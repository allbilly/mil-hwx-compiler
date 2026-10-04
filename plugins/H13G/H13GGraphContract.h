#import <Foundation/Foundation.h>
#import "ANEGraphIR.h"

NS_ASSUME_NONNULL_BEGIN

// A name-independent, typed SSA contract. Literal constants and every operand
// edge participate; BLOBFILE paths/offsets are deliberately not target choices.
NSDictionary *H13GGraphContract(ANEGraphFunction *function);

NS_ASSUME_NONNULL_END
