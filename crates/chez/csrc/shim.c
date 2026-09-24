#include "shim.h"

ptr chez_nil(void) { return Snil; }
ptr chez_true(void) { return Strue; }
ptr chez_false(void) { return Sfalse; }
ptr chez_void(void) { return Svoid; }
ptr chez_eof_object(void) { return Seof_object; }
ptr chez_char(unsigned int c) { return Schar(c); }

int chez_fixnump(ptr x) { return Sfixnump(x); }
int chez_charp(ptr x) { return Scharp(x); }
int chez_nullp(ptr x) { return Snullp(x); }
int chez_eof_objectp(ptr x) { return Seof_objectp(x); }
int chez_booleanp(ptr x) { return Sbooleanp(x); }
int chez_pairp(ptr x) { return Spairp(x); }
int chez_symbolp(ptr x) { return Ssymbolp(x); }
int chez_procedurep(ptr x) { return Sprocedurep(x); }
int chez_flonump(ptr x) { return Sflonump(x); }
int chez_vectorp(ptr x) { return Svectorp(x); }
int chez_bytevectorp(ptr x) { return Sbytevectorp(x); }
int chez_stringp(ptr x) { return Sstringp(x); }
int chez_bignump(ptr x) { return Sbignump(x); }

iptr chez_fixnum_value(ptr x) { return Sfixnum_value(x); }
unsigned int chez_char_value(ptr x) { return Schar_value(x); }
int chez_boolean_value(ptr x) { return Sboolean_value(x); }
double chez_flonum_value(ptr x) { return Sflonum_value(x); }
ptr chez_car(ptr x) { return Scar(x); }
ptr chez_cdr(ptr x) { return Scdr(x); }
iptr chez_string_length(ptr x) { return Sstring_length(x); }
unsigned int chez_string_ref(ptr x, iptr i) { return Sstring_ref(x, i); }
iptr chez_vector_length(ptr x) { return Svector_length(x); }
ptr chez_vector_ref(ptr x, iptr i) { return Svector_ref(x, i); }
iptr chez_bytevector_length(ptr x) { return Sbytevector_length(x); }
octet *chez_bytevector_data(ptr x) { return Sbytevector_data(x); }
