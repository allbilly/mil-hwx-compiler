#import "H13GProgramEncoder.h"
#import "H13GObjectWriter.h"
#import "../H13GGraphContract.h"
#import "ANEBlobResolver.h"
#import <CommonCrypto/CommonDigest.h>

#include "H13GConstantPacker.h"
#include "H13GTaskEncoder.h"
#include "H13GTargetData.inc"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

static void failure(ANEDiagnosticEngine *diagnostics, ANEGraphFunction *function,
                    NSString *code, NSString *message) {
    ANESourceLocation location = ANESourceLocationMake(0, 1, 1);
    ANESourceRange range = function.operations.count ? function.operations[0].range
        : ANESourceRangeMake(location, location);
    [diagnostics emitSeverity:ANEDiagnosticSeverityError code:code message:message range:range];
}

static NSData *constant(NSArray<ANEGraphValue *> *values, NSUInteger identifier,
                         NSUInteger bytes, NSURL *root, ANEDiagnosticEngine *diagnostics) {
    return [ANEBlobResolver loadConstantForOperation:values[identifier].producer
        expectedBytes:bytes modelRoot:root diagnostics:diagnostics];
}

static std::vector<uint16_t> halfWords(NSData *data) {
    std::vector<uint16_t> words(data.length / 2);
    memcpy(words.data(), data.bytes, data.length);
    return words;
}

static float halfValue(uint16_t bits) {
    _Float16 value;
    memcpy(&value, &bits, sizeof(bits));
    return static_cast<float>(value);
}

// The captured PE affine path loses precision for small folded biases. Select
// the smallest power-of-two scale that brings nonzero beta/gamma to 2^-16.
// The NE path compensates using AccBias's right shift (32-log2(scale)).
static unsigned affineExponent(const std::vector<uint16_t>& gamma,
                               const std::vector<uint16_t>& beta) {
    float minimum = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < gamma.size(); ++i) {
        float g = halfValue(gamma[i]), b = halfValue(beta[i]);
        if (!std::isfinite(g) || !std::isfinite(b) || g == 0)
            throw std::invalid_argument("H13G affine requires finite coefficients and nonzero gamma");
        const float ratio = std::abs(b / g);
        if (ratio != 0) minimum = std::min(minimum, ratio);
    }
    unsigned exponent = 0;
    while (minimum < std::ldexp(1.0f, -16)) {
        minimum *= 2;
        if (++exponent > 15)
            throw std::invalid_argument("H13G affine requires an unmeasured bias scale");
    }
    return exponent;
}

static NSData *wordData(NSArray<NSNumber *> *words) {
    NSMutableData *data = [NSMutableData dataWithLength:words.count * 2];
    uint16_t *destination = (uint16_t *)data.mutableBytes;
    for (NSUInteger i = 0; i < words.count; ++i) destination[i] = words[i].unsignedShortValue;
    return data;
}

static void place(NSMutableData *region, NSUInteger offset, NSData *data) {
    if (offset > region.length || data.length > region.length - offset)
        throw std::invalid_argument("H13G coefficient placement exceeds the planned region");
    [region replaceBytesInRange:NSMakeRange(offset, data.length) withBytes:data.bytes];
}

@implementation H13GProgramEncoder
+ (ANEHWXArtifact *)encodeFunction:(ANEGraphFunction *)function
                         modelRoot:(NSURL *)modelRoot
                      objectLabels:(NSDictionary<NSString *, NSString *> *)objectLabels
                       diagnostics:(ANEDiagnosticEngine *)diagnostics {
    NSData *targetData = [NSData dataWithBytes:kH13GTargetData length:strlen(kH13GTargetData)];
    NSDictionary *target = [NSJSONSerialization JSONObjectWithData:targetData options:0 error:nil];
    NSDictionary *contract = H13GGraphContract(function);
    NSString *contractName = nil;
    for (NSString *key in target[@"contracts"])
        if ([contract isEqual:target[@"contracts"][key]]) { contractName = key; break; }
    if (!contractName) {
        failure(diagnostics, function, @"h13g.legalize.unsupported-graph",
            @"H13G has no measured schedule for this typed graph, geometry, constants and operand edges");
        return nil;
    }
    NSMutableArray<ANEGraphValue *> *values = [function.inputs mutableCopy];
    for (ANEGraphOperation *operation in function.operations) [values addObject:operation.result];
    NSDictionary *plan = target[@"plans"][contractName];
    NSData *gammaData = constant(values, [plan[@"gamma"] unsignedIntegerValue], 768 * 2, modelRoot, diagnostics);
    NSData *betaData = constant(values, [plan[@"beta"] unsignedIntegerValue], 768 * 2, modelRoot, diagnostics);
    if (!gammaData || !betaData) return nil;
    try {
        auto gamma = halfWords(gammaData), beta = halfWords(betaData);
        unsigned exponent = 0;
        if ([contractName isEqualToString:@"norm_expand_gelu_project_residual"]) {
            exponent = affineExponent(gamma, beta);
            if (exponent) plan = target[@"plans"][[contractName stringByAppendingString:@"_engine_affine"]];
        } else {
            // Also reject invalid coefficients on measured linear affine paths.
            (void)affineExponent(gamma, beta);
        }
        BOOL engineAffine = [plan[@"engine_affine"] boolValue];
        auto affine = h13g::packAffine(gamma, beta,
            engineAffine ? h13g::AffineLayout::EnginePairs : h13g::AffineLayout::Linear,
            std::ldexp(1.0f, (int)exponent));
        NSData *affineData = [NSData dataWithBytes:affine.data() length:affine.size()];
        NSDictionary *object = plan[@"object"];
        NSUInteger coefficientBytes = 0, constantBytes = 0;
        for (NSDictionary *command in object[@"commands"]) {
            for (NSDictionary *section in command[@"sections"]) {
                if ([section[@"segment"] isEqualToString:@"__KERN_0"])
                    coefficientBytes = [section[@"size"] unsignedIntegerValue];
                if ([section[@"segment"] isEqualToString:@"__TEXT"] &&
                    [section[@"name"] isEqualToString:@"__const"])
                    constantBytes = [section[@"size"] unsignedIntegerValue];
            }
        }
        NSMutableData *coefficients = [NSMutableData dataWithLength:coefficientBytes];
        NSMutableData *constants = [NSMutableData dataWithLength:constantBytes];
        place(coefficients, 0, wordData(plan[@"lut"]));
        if (engineAffine) place(coefficients, 256, affineData);
        else place(constants, 0, affineData);
        for (NSDictionary *matrix in plan[@"matrices"]) {
            NSUInteger inputs = [matrix[@"inputs"] unsignedIntegerValue];
            NSUInteger outputs = [matrix[@"outputs"] unsignedIntegerValue];
            NSData *w = constant(values, [matrix[@"weight"] unsignedIntegerValue],
                                 inputs * outputs * 2, modelRoot, diagnostics);
            NSData *b = constant(values, [matrix[@"bias"] unsignedIntegerValue],
                                 outputs * 2, modelRoot, diagnostics);
            if (!w || !b) return nil;
            std::vector<std::size_t> tiles;
            for (NSNumber *tile in matrix[@"tiles"]) tiles.push_back(tile.unsignedIntegerValue);
            auto packed = h13g::packMatrix(halfWords(w), halfWords(b), inputs, outputs, tiles);
            place(coefficients, [matrix[@"offset"] unsignedIntegerValue],
                  [NSData dataWithBytes:packed.data() length:packed.size()]);
        }
        if (plan[@"gelu"])
            place(coefficients, [plan[@"gelu"][@"offset"] unsignedIntegerValue],
                  wordData(plan[@"gelu"][@"values"]));
        if (plan[@"mask"]) {
            NSData *mask = constant(values, [plan[@"mask"] unsignedIntegerValue], 32 * 32 * 2, modelRoot, diagnostics);
            if (!mask) return nil;
            place(constants, affineData.length, mask);
            NSData *expLUT = [wordData(plan[@"lut"]) subdataWithRange:NSMakeRange(128, 128)];
            place(coefficients, [plan[@"softmax_lut_offset"] unsignedIntegerValue], expLUT);
        }
        std::vector<h13g::Task> tasks;
        NSArray *packets = target[@"packets"];
        for (NSDictionary *row in plan[@"tasks"]) {
            h13g::Task task;
            for (NSUInteger i = 0; i < 10; ++i) task.controls[i] = [row[@"controls"][i] unsignedIntValue];
            task.extendedControl = [row[@"extended"] unsignedIntValue];
            task.paddingAfter = [row[@"padding"] unsignedIntegerValue];
            for (NSNumber *packetIndex in row[@"packets"]) {
                NSDictionary *packetRow = packets[packetIndex.unsignedIntegerValue];
                h13g::RegisterPacket packet{[packetRow[@"address"] unsignedIntValue], {}};
                for (NSNumber *value in packetRow[@"values"]) packet.values.push_back(value.unsignedIntValue);
                if (engineAffine && tasks.size() == 8 && packet.address == 0xc800)
                    packet.values[3] = (32 - exponent) << 16;
                task.packets.push_back(std::move(packet));
            }
            tasks.push_back(std::move(task));
        }
        auto program = h13g::encodeTasks(tasks);
        h13g::decodeTasks(program.bytes, program.sizes[0], tasks.size());
        NSData *taskData = [NSData dataWithBytes:program.bytes.data() length:program.bytes.size()];
        NSMutableDictionary<NSString *, NSString *> *symbolAliases = [NSMutableDictionary dictionary];
        NSString *planName = engineAffine ? [contractName stringByAppendingString:@"_engine_affine"] : contractName;
        for (NSDictionary *group in plan[@"kernel_groups"]) {
            NSUInteger offset = [group[@"offset"] unsignedIntegerValue];
            NSUInteger length = [group[@"size"] unsignedIntegerValue];
            unsigned char digest[CC_SHA256_DIGEST_LENGTH];
            CC_SHA256((const uint8_t *)coefficients.bytes + offset, (CC_LONG)length, digest);
            NSMutableString *hash = [NSMutableString string];
            for (auto byte : digest) [hash appendFormat:@"%02x", byte];
            NSString *key = [NSString stringWithFormat:@"%@:%lu:%lu:%@", planName,
                (unsigned long)offset, (unsigned long)length, hash];
            symbolAliases[group[@"prefix"]] = target[@"symbol_aliases"][key] ?:
                [@"K" stringByAppendingString:hash.uppercaseString];
        }
        NSDictionary *labels = @{
            @"source-label": objectLabels[@"source-label"] ?: [modelRoot.path stringByAppendingPathComponent:@"model.mil"],
            @"output-label": objectLabels[@"output-label"] ?: @"program-0.hwx"};
        NSError *error = nil;
        NSData *image = [H13GObjectWriter buildObjectWithLayout:object values:values
            objectLabels:labels symbolAliases:symbolAliases
            taskDescriptor:taskData firstTaskSize:program.sizes[0] taskCount:tasks.size()
            constantRegion:constants coefficientRegion:coefficients error:&error];
        if (!image) {
            failure(diagnostics, function, @"h13g.encode.invalid-object",
                    error.localizedDescription ?: @"cannot construct H13G object");
            return nil;
        }
        NSMutableArray<ANEHWXBinding *> *bindings = [NSMutableArray array];
        for (NSDictionary *command in object[@"commands"]) {
            if ([command[@"cmd"] unsignedIntegerValue] != 4 ||
                [command[@"flavor"] unsignedIntegerValue] != 3) continue;
            ANEGraphValue *value = values[[command[@"value"] unsignedIntegerValue]];
            BOOL output = [command[@"output"] boolValue];
            NSUInteger row = (value.type.shape[3].unsignedIntegerValue * 2 + 63) / 64 * 64;
            NSUInteger plane = value.type.shape[2].unsignedIntegerValue * row;
            NSUInteger batch = value.type.shape[1].unsignedIntegerValue * plane;
            NSUInteger bytes = value.type.shape[0].unsignedIntegerValue * batch;
            [bindings addObject:[[ANEHWXBinding alloc] initWithIdentifier:value.name
                role:output ? ANESurfaceRoleOutput : ANESurfaceRoleInput
                logicalByteLength:bytes allocationByteLength:(bytes + 0x3fff) / 0x4000 * 0x4000
                ioSurfaceIndex:[command[@"index"] integerValue] - 1
                rowStrideBytes:row planeStrideBytes:plane batchStrideBytes:batch]];
        }
        NSMutableArray *operations = [NSMutableArray array];
        for (ANEGraphOperation *operation in function.operations)
            if (![operation.operationName isEqualToString:@"const"])
                [operations addObject:operation.operationName];
        return [[ANEHWXArtifact alloc] initWithImage:image bindings:bindings operations:operations];
    } catch (const std::exception& error) {
        failure(diagnostics, function, @"h13g.encode.invalid-plan",
                [NSString stringWithUTF8String:error.what()]);
        return nil;
    }
}
@end
