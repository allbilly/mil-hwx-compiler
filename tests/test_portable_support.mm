#import <Foundation/Foundation.h>
#import "ANEJSONSerialization.h"
#import "ANEMachO.h"
#import "ANESHA256.h"

#include <stdio.h>

int main(void) {
    @autoreleasepool {
        unsigned char digest[ANE_SHA256_DIGEST_LENGTH];
        ANESHA256("abc", 3, digest);
        NSMutableString *hash = [NSMutableString string];
        for (unsigned char byte : digest) [hash appendFormat:@"%02x", byte];
        if (![hash isEqualToString:
            @"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"]) {
            fprintf(stderr, "SHA-256 known vector failed\n");
            return 1;
        }
        NSDictionary *object = @{
            @"z": @[@3, @YES, @NO, [NSNull null]],
            @"a": @{@"z": @"quote\" slash\\ newline\n雪", @"a": @-1.25},
        };
        NSDictionary *reordered = @{
            @"a": @{@"a": @-1.25, @"z": object[@"a"][@"z"]},
            @"z": object[@"z"],
        };
        for (NSNumber *pretty in @[@NO, @YES]) {
            NSError *error = nil;
            NSData *data = ANEJSONData(object, pretty.boolValue, &error);
            NSData *repeat = ANEJSONData(reordered, pretty.boolValue, &error);
            id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
            if (!data || error || ![parsed isEqual:object] || ![data isEqual:repeat] ||
                [text rangeOfString:@"\"a\""].location >
                [text rangeOfString:@"\"z\""].location) {
                fprintf(stderr, "stable JSON round trip failed\n");
                return 1;
            }
        }
        printf("portable hashes, wire layouts and JSON: PASS\n");
    }
    return 0;
}
