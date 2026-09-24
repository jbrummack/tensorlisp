/* Function wrappers for the scheme.h macros. The macros hardcode per-machine
 * tag bits and field offsets, so they are compiled against the installed header
 * instead of being reimplemented in Rust. */
#include "scheme.h"

ptr chez_nil(void);
ptr chez_true(void);
ptr chez_false(void);
ptr chez_void(void);
ptr chez_eof_object(void);
ptr chez_char(unsigned int c);

int chez_fixnump(ptr x);
int chez_charp(ptr x);
int chez_nullp(ptr x);
int chez_eof_objectp(ptr x);
int chez_booleanp(ptr x);
int chez_pairp(ptr x);
int chez_symbolp(ptr x);
int chez_procedurep(ptr x);
int chez_flonump(ptr x);
int chez_vectorp(ptr x);
int chez_bytevectorp(ptr x);
int chez_stringp(ptr x);
int chez_bignump(ptr x);

iptr chez_fixnum_value(ptr x);
unsigned int chez_char_value(ptr x);
int chez_boolean_value(ptr x);
double chez_flonum_value(ptr x);
ptr chez_car(ptr x);
ptr chez_cdr(ptr x);
iptr chez_string_length(ptr x);
unsigned int chez_string_ref(ptr x, iptr i);
iptr chez_vector_length(ptr x);
ptr chez_vector_ref(ptr x, iptr i);
iptr chez_bytevector_length(ptr x);
octet *chez_bytevector_data(ptr x);
