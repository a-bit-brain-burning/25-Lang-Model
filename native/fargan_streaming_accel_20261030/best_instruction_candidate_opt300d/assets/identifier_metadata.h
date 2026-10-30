#ifndef IDENTIFIER_METADATA_H
#define IDENTIFIER_METADATA_H
typedef struct {unsigned group,start,end,symbol_index,connect_previous,first_digit,tone; char symbol;} IdentifierUnit;
static const unsigned identifier_unit_count=0;
static const IdentifierUnit identifier_units[1]={
{0,0,0,0,0,0,0,0},
};
static const unsigned identifier_group_count=0;
static const unsigned char identifier_rhythm_core[34]={1,0,1,1,0,1,1,0,1,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,1,1,1,0,1,1,0,1,1};
#endif
