/* Copyright (c) 2018-2019 Mozilla
                 2023 Amazon */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#define RTCD_ARCH c

#include "nnet_arch.h"

unsigned long long bwe_mve_vectors,bwe_scalar_macs,bwe_i8_dots,bwe_i8_inputs;

unsigned long long bwe_quant_groups;

unsigned long long bwe_activation_calls[8],bwe_activation_elements[8];
unsigned long long bwe_act_rational_vectors,bwe_act_scalar_divisions,bwe_act_scalar_tail_elements,bwe_act_vector_operations,bwe_act_vector_loads,bwe_act_vector_stores;

unsigned long long bwe_act_guard_vectors,bwe_act_fallback_vectors,bwe_act_fallback_elements;

unsigned long long bwe_exp_guard_vectors,bwe_exp_fast_vectors,bwe_exp_fallback_vectors,bwe_exp_scalar_elements;
unsigned long long bwe_exp_vector_arithmetic_operations,bwe_exp_guard_operations,bwe_exp_explicit_vector_loads,bwe_exp_explicit_vector_stores;
