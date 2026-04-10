// External scanner for shadml tree-sitter grammar.
//
// Emits two layout tokens:
//   LAYOUT_END       — after a newline, when the next non-blank line starts
//                      at column 0 (top-level boundary) or at EOF.
//   LAYOUT_SEMICOLON — after a newline, when the next non-blank line looks
//                      like a new binding (lowercase identifier followed
//                      eventually by '=' that isn't '==').

#include "tree_sitter/parser.h"
#include <stdbool.h>

enum TokenType {
  LAYOUT_END,
  LAYOUT_SEMICOLON,
};

void *tree_sitter_shadml_external_scanner_create(void) { return NULL; }
void tree_sitter_shadml_external_scanner_destroy(void *p) {}
unsigned tree_sitter_shadml_external_scanner_serialize(void *p, char *b) {
  return 0;
}
void tree_sitter_shadml_external_scanner_deserialize(void *p, const char *b,
                                                     unsigned n) {}

static bool is_ident_start(int32_t c) {
  return (c >= 'a' && c <= 'z') || c == '_';
}

static bool is_ident_continue(int32_t c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '\'';
}

bool tree_sitter_shadml_external_scanner_scan(void *payload, TSLexer *lexer,
                                              const bool *valid_symbols) {
  bool want_end = valid_symbols[LAYOUT_END];
  bool want_semi = valid_symbols[LAYOUT_SEMICOLON];
  if (!want_end && !want_semi) return false;

  // Skip whitespace, tracking newlines.
  bool saw_newline = false;

  while (lexer->lookahead == ' ' || lexer->lookahead == '\t' ||
         lexer->lookahead == '\r' || lexer->lookahead == '\n') {
    if (lexer->lookahead == '\n') {
      saw_newline = true;
    }
    lexer->advance(lexer, true);  // skip whitespace
  }

  if (!saw_newline) return false;

  uint32_t next_col = lexer->get_column(lexer);
  bool at_eof = lexer->eof(lexer);

  // LAYOUT_END: column 0 or EOF.
  if (want_end && (at_eof || next_col == 0)) {
    lexer->result_symbol = LAYOUT_END;
    return true;
  }

  // LAYOUT_SEMICOLON: lookahead to check if this line starts a new binding.
  // A binding starts with a lowercase identifier eventually followed by '='
  // (which isn't '==') on the same line.
  if (want_semi && !at_eof && next_col > 0 && is_ident_start(lexer->lookahead)) {
    // Mark the end of the token HERE (zero-width, after whitespace).
    // All further advances are just lookahead — they won't be part of the token.
    lexer->mark_end(lexer);

    // Skip the identifier.
    while (is_ident_continue(lexer->lookahead)) {
      lexer->advance(lexer, false);
    }

    // Now scan forward on the same line for '=' (not '==').
    int depth = 0;
    for (int i = 0; i < 200; i++) {
      int32_t c = lexer->lookahead;
      if (c == '\n' || c == '\r' || c == 0) break;

      if (c == '=' && depth == 0) {
        lexer->advance(lexer, false);
        if (lexer->lookahead != '=') {
          // Found a bare '=' — this is a new binding.
          lexer->result_symbol = LAYOUT_SEMICOLON;
          return true;
        }
        // It was '==', which is a comparison, not a binding.
        break;
      }

      // Track parentheses depth to skip '=' inside parens.
      if (c == '(') depth++;
      if (c == ')') {
        if (depth > 0) depth--;
        else break;  // unbalanced — not a binding
      }

      // Stop on tokens that indicate this isn't a binding pattern.
      if (c == '-') {
        // Could be '->' which means type signature or lambda, not binding.
        lexer->advance(lexer, false);
        if (lexer->lookahead == '>') break;
        continue;
      }
      if (c == ':') break;  // type annotation — it's a type_signature, not binding

      lexer->advance(lexer, false);
    }
  }

  return false;
}
