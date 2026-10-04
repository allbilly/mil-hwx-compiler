#import <Foundation/Foundation.h>
#import "ANEGraphIR.h"
#import "ANEHWXArtifact.h"

NS_ASSUME_NONNULL_BEGIN

@interface H13GProgramEncoder : NSObject
+ (nullable ANEHWXArtifact *)encodeFunction:(ANEGraphFunction *)function
                                 modelRoot:(NSURL *)modelRoot
                              objectLabels:(nullable NSDictionary<NSString *, NSString *> *)objectLabels
                               diagnostics:(ANEDiagnosticEngine *)diagnostics;
@end

NS_ASSUME_NONNULL_END
