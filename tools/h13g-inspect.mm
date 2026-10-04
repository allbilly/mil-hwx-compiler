#import <Foundation/Foundation.h>
#import "ANEDiagnostic.h"
#import "ANEGraphVerifier.h"
#import "H13GGraphContract.h"
#import "MILLexer.h"
#import "MILParser.h"
#import "MILGraphImporter.h"

#include <cstdio>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 64;
        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:argv[1]]];
        if (!data) return 66;
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        MILLexer *lexer = [[MILLexer alloc] initWithData:data diagnostics:diagnostics];
        MILParser *parser = [[MILParser alloc] initWithTokens:lexer.lexAllTokens
                                               diagnostics:diagnostics];
        MILProgramSyntax *syntax = parser.parseProgram;
        ANEGraphModule *module = syntax ? [MILGraphImporter importProgram:syntax
            diagnostics:diagnostics] : nil;
        if (!module || module.functions.count != 1 ||
            ![ANEGraphVerifier verifyModule:module diagnostics:diagnostics]) {
            for (ANEDiagnostic *d in diagnostics.diagnostics)
                fprintf(stderr, "%s: %s\n", d.code.UTF8String, d.message.UTF8String);
            return 65;
        }
        ANEGraphFunction *function = module.functions[0];
        NSMutableArray *names = [NSMutableArray array];
        for (ANEGraphValue *input in function.inputs) [names addObject:input.name];
        for (ANEGraphOperation *op in function.operations) [names addObject:op.result.name];
        NSData *json = [NSJSONSerialization dataWithJSONObject:
            @{@"contract": H13GGraphContract(function), @"names": names}
            options:NSJSONWritingSortedKeys error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        putchar('\n');
        return 0;
    }
}
