#import <Foundation/Foundation.h>
#import "ANEGraphIR.h"

NS_ASSUME_NONNULL_BEGIN

@interface H13GObjectWriter : NSObject
+ (nullable NSData *)buildObjectWithLayout:(NSDictionary *)layout
                                    values:(NSArray<ANEGraphValue *> *)values
                                objectLabels:(NSDictionary<NSString *, NSString *> *)objectLabels
                               symbolAliases:(NSDictionary<NSString *, NSString *> *)symbolAliases
                              taskDescriptor:(NSData *)tasks
                               firstTaskSize:(NSUInteger)firstTaskSize
                                   taskCount:(NSUInteger)taskCount
                              constantRegion:(NSData *)constants
                           coefficientRegion:(NSData *)coefficients
                                       error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
