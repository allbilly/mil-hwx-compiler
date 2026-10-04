#import "H13GGraphContract.h"

static NSArray *valueType(ANEValueType *type) {
    return @[@(type.kind), @(type.elementType), type.shape];
}

static id argument(ANEGraphArgument *arg, NSDictionary *identifiers) {
    if (arg.kind == ANEGraphArgumentKindValue)
        return @{@"value": identifiers[arg.value.name]};
    // Only the declaration's tensor type participates for learned constants.
    // The resolver validates the actual file reference when loading weights.
    if ([arg.calleeName isEqualToString:@"BLOBFILE"])
        return @{@"blob": @YES};
    NSMutableArray *children = [NSMutableArray array];
    for (ANEGraphNamedArgument *child in arg.callArguments)
        [children addObject:@[child.name ?: @"", argument(child.value, identifiers)]];
    NSMutableArray *elements = [NSMutableArray array];
    for (ANEGraphArgument *child in arg.elements)
        [elements addObject:argument(child, identifiers)];
    // Numeric spelling (1e-05 vs 0.00001) does not affect the contract.
    id atom = arg.text ?: @"";
    if (arg.kind == ANEGraphArgumentKindInteger ||
        arg.kind == ANEGraphArgumentKindFloatingPoint)
        atom = @([arg.text doubleValue]);
    return @{@"kind": @(arg.kind), @"atom": atom,
             @"callee": arg.calleeName ?: @"",
             @"type": arg.calleeValueType ? valueType(arg.calleeValueType) : @[],
             @"arguments": children, @"elements": elements};
}

static NSDictionary *arguments(NSDictionary<NSString *, ANEGraphArgument *> *args,
                               NSDictionary *identifiers) {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (NSString *key in args) {
        if ([key isEqualToString:@"name"]) continue;
        result[key] = argument(args[key], identifiers);
    }
    return result;
}

NSDictionary *H13GGraphContract(ANEGraphFunction *function) {
    NSMutableDictionary *identifiers = [NSMutableDictionary dictionary];
    NSMutableArray *inputs = [NSMutableArray array];
    for (ANEGraphValue *input in function.inputs) {
        identifiers[input.name] = @(identifiers.count);
        [inputs addObject:valueType(input.type)];
    }
    NSMutableArray *operations = [NSMutableArray array];
    for (ANEGraphOperation *operation in function.operations) {
        identifiers[operation.result.name] = @(identifiers.count);
        [operations addObject:@{@"op": operation.operationName,
            @"type": valueType(operation.result.type),
            @"arguments": arguments(operation.arguments, identifiers),
            @"attributes": arguments(operation.attributes, identifiers)}];
    }
    NSMutableArray *returns = [NSMutableArray array];
    for (ANEGraphValue *value in function.returnValues)
        [returns addObject:identifiers[value.name]];
    return @{@"inputs": inputs, @"operations": operations, @"returns": returns};
}
