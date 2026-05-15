#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

double lempel_ziv_complexity(const uint8_t* binary_sequence, int length) {
    int cn = 1;
    double bn;
    const uint8_t* end_of_sequence = binary_sequence + length;
    const uint8_t* s = binary_sequence;
    int s_length = 1;
    const uint8_t* q = binary_sequence + s_length;
    int q_length = 1;

    while ((q + q_length) <= end_of_sequence) {
        int sqpi_length = s_length + q_length - 1;
        void* found = memmem(s, sqpi_length, q, q_length);
        if (found) {
            ++q_length;
        } else {
            ++cn;
            s_length += q_length;
            q += q_length;
            q_length = 1;
        }
    }

    bn = (double)length / log2(length);
    return (double)cn / bn;
}

double imf_lempel_ziv_complexity(const double* imf, int length) {
    double lz = 0.0;
    int bin_len = length * 12;
    uint8_t* binary_sequence = (uint8_t*)malloc((size_t)bin_len * sizeof(uint8_t));
    int i;
    int j;

    if (binary_sequence == NULL) {
        return 0.0;
    }

    for (i = 0, j = 11; i < length; ++i, j += 12) {
        int c;
        uint16_t value = (uint16_t)imf[i];
        for (c = 0; c < 12; ++c) {
            binary_sequence[j - c] = (uint8_t)((value >> c) & 1);
        }
    }

    lz = lempel_ziv_complexity(binary_sequence, bin_len);
    free(binary_sequence);
    return lz;
}