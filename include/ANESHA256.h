#pragma once

#include <stddef.h>

// Keep coefficient IDs and verification hashes identical on both hosts.
#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>
#define ANE_SHA256_DIGEST_LENGTH CC_SHA256_DIGEST_LENGTH
static inline unsigned char *ANESHA256(const void *bytes, size_t length,
                                      unsigned char *digest) {
    return CC_SHA256(bytes, (CC_LONG)length, digest);
}
#else
#include <openssl/sha.h>
#define ANE_SHA256_DIGEST_LENGTH SHA256_DIGEST_LENGTH
static inline unsigned char *ANESHA256(const void *bytes, size_t length,
                                      unsigned char *digest) {
    return SHA256((const unsigned char *)bytes, length, digest);
}
#endif
