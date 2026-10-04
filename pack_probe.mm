#import <Foundation/Foundation.h>
#import "H16GConstantPacker.h"
#include <stdio.h>
#include <stdlib.h>

// Calls the unmodified upstream implementation; no packing logic lives here.
int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 6) return 64;
        NSString *input = [NSString stringWithUTF8String:argv[1]];
        NSString *output = [NSString stringWithUTF8String:argv[2]];
        NSData *raw = [NSData dataWithContentsOfFile:input];
        if (!raw) return 66;
        NSError *error = nil;
        NSData *packed = [H16GConstantPacker packConv1x1Weights:raw
            inputChannels:(NSUInteger)strtoul(argv[3], NULL, 10)
            outputChannels:(NSUInteger)strtoul(argv[4], NULL, 10)
            bytesPerWeight:2
            packingFormat:(H16GConvWeightPackingFormat)strtoul(argv[5], NULL, 10)
            error:&error];
        if (!packed) {
            fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
            return 65;
        }
        return [packed writeToFile:output atomically:YES] ? 0 : 74;
    }
}
