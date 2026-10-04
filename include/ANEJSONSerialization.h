#pragma once

#import <Foundation/Foundation.h>

#ifdef GNUSTEP
// GNUstep Base 1.31 has no NSJSONWritingSortedKeys. Write container structure
// in a stable order and delegate scalar escaping/number formatting to Base.
static inline NSString *ANEJSONText(id value, BOOL pretty, NSUInteger depth) {
    BOOL dictionary = [value isKindOfClass:[NSDictionary class]];
    BOOL array = [value isKindOfClass:[NSArray class]];
    if (!dictionary && !array) {
        NSData *encoded = [NSJSONSerialization dataWithJSONObject:@[value]
            options:0 error:nullptr];
        NSString *text = [[NSString alloc] initWithData:encoded
                                             encoding:NSUTF8StringEncoding];
        return [text substringWithRange:NSMakeRange(1, text.length - 2)];
    }
    NSArray *keys = dictionary
        ? [[value allKeys] sortedArrayUsingSelector:@selector(compare:)] : value;
    NSMutableArray<NSString *> *entries = [NSMutableArray array];
    NSString *indent = [@"" stringByPaddingToLength:(depth + 1) * 2
        withString:@" " startingAtIndex:0];
    for (id key in keys) {
        NSString *entry = dictionary
            ? [NSString stringWithFormat:@"%@:%@%@", ANEJSONText(key, NO, 0),
                pretty ? @" " : @"", ANEJSONText(value[key], pretty, depth + 1)]
            : ANEJSONText(key, pretty, depth + 1);
        [entries addObject:pretty ? [indent stringByAppendingString:entry] : entry];
    }
    NSString *open = dictionary ? @"{" : @"[";
    NSString *close = dictionary ? @"}" : @"]";
    if (entries.count == 0) return [open stringByAppendingString:close];
    if (!pretty) return [NSString stringWithFormat:@"%@%@%@", open,
        [entries componentsJoinedByString:@","], close];
    NSString *outerIndent = [@"" stringByPaddingToLength:depth * 2
        withString:@" " startingAtIndex:0];
    return [NSString stringWithFormat:@"%@\n%@\n%@%@", open,
        [entries componentsJoinedByString:@",\n"], outerIndent, close];
}
#endif

static inline NSData *ANEJSONData(id object, BOOL pretty, NSError **error) {
#ifdef GNUSTEP
    // Validate first so malformed input receives the Foundation error path.
    NSData *validated = [NSJSONSerialization dataWithJSONObject:object
        options:0 error:error];
    if (!validated) return nil;
    return [ANEJSONText(object, pretty, 0) dataUsingEncoding:NSUTF8StringEncoding];
#else
    return [NSJSONSerialization dataWithJSONObject:object
        options:NSJSONWritingSortedKeys | (pretty ? NSJSONWritingPrettyPrinted : 0)
        error:error];
#endif
}
