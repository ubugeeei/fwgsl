; Zed highlight queries for fwgsl
; These are shared with the tree-sitter-fwgsl grammar.
; See ../../tree-sitter-fwgsl/queries/highlights.scm for the canonical version.

; Keywords — control flow
["if" "then" "else" "case" "of" "match" "loop"] @keyword

; Keywords — storage/declaration
["let" "in" "const" "data" "alias" "extern" "uniform" "storage"] @keyword

; Keywords — other
["module" "import" "where" "trait" "impl" "bitfield"
 "do" "forall" "deriving" "as" "when" "cfg" "not"] @keyword

; Booleans
(boolean) @boolean

; Attributes
(attribute "@" @punctuation.special (identifier) @attribute)

; Types (UpperIdent)
(type_constructor) @type
(type_application (upper_identifier) @type)
(upper_identifier) @type

; Function definitions
(function_declaration name: (identifier) @function)
(type_signature name: (identifier) @function)

; Binding declarations (GPU resources)
(binding_declaration name: (identifier) @variable)
(binding_declaration "group" @attribute)
(binding_declaration "binding" @attribute)

; Builtin functions ($prefixed)
(builtin_identifier) @function.builtin

; Constructor patterns
(constructor_pattern (upper_identifier) @type)

; Record fields
(record_field name: (identifier) @property)
(field_init name: (identifier) @property)
(field_access "." (identifier) @property)

; Variables
(identifier_expression) @variable

; Numeric literals
(integer_literal) @number
(float_literal) @number

; Strings
(string_literal) @string
(char_literal) @string
(escape_sequence) @string.escape

; Comments
(line_comment) @comment
(block_comment) @comment

; Operators
["+" "-" "*" "/" "%" "==" "/=" "<" ">" "<=" ">=" "&&" "||"] @operator
["|>" "$" "->" "=>" "=" "|" "\\" ":" "::" "."] @operator

; Punctuation
["(" ")" "[" "]" "{" "}"] @punctuation.bracket
["," ";"] @punctuation.delimiter
["@"] @punctuation.special
