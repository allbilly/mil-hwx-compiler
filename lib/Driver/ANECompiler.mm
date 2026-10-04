#import "ANECompiler.h"

#import "ANEExecutableBundle.h"
#import "ANEStagedCompiler.h"

@implementation ANECompiler
- (ANEExecutableBundle *)compileMILData:(NSData *)milData
                              modelRoot:(NSURL *)modelRoot
                                 target:(NSString *)target
                           objectLabels:(NSDictionary<NSString *, NSString *> *)objectLabels
                            diagnostics:(ANEDiagnosticEngine *)diagnostics {
    return [ANEStagedCompiler compileMILData:milData modelRoot:modelRoot
        target:target objectLabels:objectLabels diagnostics:diagnostics];
}
- (ANEExecutableBundle *)compileMILData:(NSData *)milData
                              modelRoot:(NSURL *)modelRoot
                                 target:(NSString *)target
                            diagnostics:(ANEDiagnosticEngine *)diagnostics {
    return [ANEStagedCompiler compileMILData:milData modelRoot:modelRoot
        target:target objectLabels:nil diagnostics:diagnostics];
}
@end
