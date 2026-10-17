//! Paren checking for the Scheme that models write.
//!
//! Chez reports a reader error as `unexpected close parenthesis at char 1528`, which a model cannot act
//! on. [`explain`] turns it into a line and column and, when the parentheses do not balance, says which one
//! is wrong: the stray close, or the form that is never closed.
use regex::Regex;

#[derive(Debug, PartialEq, Eq)]
pub enum Problem {
    /// A `)` or `]` with nothing open.
    ExtraClose { line: usize, col: usize },
    /// A close of the wrong kind.
    Mismatch { open: (char, usize, usize), close: (char, usize, usize) },
    /// Opens left at the end: (outermost, innermost) as (line, column).
    Unclosed { outermost: (usize, usize), innermost: (usize, usize), count: usize },
}

/// Scans Scheme source (strings, `;` and `#| |#` comments and `#\(` characters are skipped) for the first
/// parenthesis problem. Lines and columns are 1-based.
pub fn check(src: &str) -> Option<Problem> {
    let c: Vec<char> = src.chars().collect();
    let (mut i, mut line, mut col) = (0usize, 1usize, 1usize);
    let mut stack: Vec<(char, usize, usize)> = Vec::new();
    let advance = |i: &mut usize, line: &mut usize, col: &mut usize| {
        if c[*i] == '\n' {
            *line += 1;
            *col = 1;
        } else {
            *col += 1;
        }
        *i += 1;
    };
    while i < c.len() {
        match c[i] {
            ';' => {
                while i < c.len() && c[i] != '\n' {
                    advance(&mut i, &mut line, &mut col);
                }
            }
            '"' => {
                advance(&mut i, &mut line, &mut col);
                while i < c.len() && c[i] != '"' {
                    if c[i] == '\\' && i + 1 < c.len() {
                        advance(&mut i, &mut line, &mut col);
                    }
                    advance(&mut i, &mut line, &mut col);
                }
                if i < c.len() {
                    advance(&mut i, &mut line, &mut col);
                }
            }
            '#' if c.get(i + 1) == Some(&'\\') => {
                // #\( #\) #\space ...: the character after the backslash is part of the literal
                for _ in 0..3.min(c.len() - i) {
                    advance(&mut i, &mut line, &mut col);
                }
                while i < c.len() && c[i].is_alphanumeric() {
                    advance(&mut i, &mut line, &mut col);
                }
            }
            '#' if c.get(i + 1) == Some(&'|') => {
                let mut depth = 0;
                while i < c.len() {
                    if c[i] == '#' && c.get(i + 1) == Some(&'|') {
                        depth += 1;
                        advance(&mut i, &mut line, &mut col);
                    } else if c[i] == '|' && c.get(i + 1) == Some(&'#') {
                        depth -= 1;
                        advance(&mut i, &mut line, &mut col);
                        if depth == 0 {
                            advance(&mut i, &mut line, &mut col);
                            break;
                        }
                    }
                    advance(&mut i, &mut line, &mut col);
                }
            }
            '(' | '[' => {
                stack.push((c[i], line, col));
                advance(&mut i, &mut line, &mut col);
            }
            ')' | ']' => {
                let close = (c[i], line, col);
                match stack.pop() {
                    None => return Some(Problem::ExtraClose { line, col }),
                    Some(open) if (open.0 == '(') != (close.0 == ')') => return Some(Problem::Mismatch { open, close }),
                    Some(_) => {}
                }
                advance(&mut i, &mut line, &mut col);
            }
            _ => advance(&mut i, &mut line, &mut col),
        }
    }
    stack.first().map(|o| Problem::Unclosed { outermost: (o.1, o.2), innermost: stack.last().map(|x| (x.1, x.2)).unwrap(), count: stack.len() })
}

fn describe(p: &Problem, src: &str) -> String {
    let text = |line: usize| src.lines().nth(line - 1).unwrap_or("").trim().chars().take(80).collect::<String>();
    match p {
        Problem::ExtraClose { line, col } => format!("an extra `)` at line {line}, column {col} closes nothing (`{}`): delete it, or a form above it closed too early", text(*line)),
        Problem::Mismatch { open: (o, ol, oc), close: (c, cl, cc) } => format!("`{o}` opened at line {ol}, column {oc} is closed by `{c}` at line {cl}, column {cc}"),
        Problem::Unclosed { outermost, innermost, count } => {
            let head = if *count == 1 { "1 parenthesis is never closed".to_string() } else { format!("{count} parentheses are never closed") };
            let tail = if outermost == innermost { String::new() } else { format!(", the innermost at line {}, column {} (`{}`)", innermost.0, innermost.1, text(innermost.0)) };
            format!("{head}: the outermost opens at line {}, column {} (`{}`){tail}; add the missing `)` at the end of the form", outermost.0, outermost.1, text(outermost.0))
        }
    }
}

/// `message` is an error from Chez about `source`; returns it with the position and the paren check added
/// where they help.
pub fn explain(message: &str, source: &str) -> String {
    let mut out = message.to_string();
    // Chez counts the character after the offender; for end-of-file it reports char 0, which says nothing
    if let Some(n) = Regex::new(r"at char (\d+)").unwrap().captures(message).and_then(|c| c[1].parse::<usize>().ok()).filter(|_| !message.contains("end-of-file")) {
        let before: String = source.chars().take(n.saturating_sub(1)).collect();
        let line = before.matches('\n').count() + 1;
        let col = before.chars().rev().take_while(|c| *c != '\n').count() + 1;
        let text = source.lines().nth(line - 1).unwrap_or("").trim();
        out.push_str(&format!(" [line {line}, column {col}: `{}`]", text.chars().take(100).collect::<String>()));
    }
    if out.contains("Exception in read") || out.contains("parenthes") || out.contains("end-of-file") {
        if let Some(p) = check(source) {
            out.push_str(&format!(" [paren check: {}]", describe(&p, source)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_source_is_fine_including_strings_comments_and_characters() {
        assert_eq!(check("(define (f x) [list x \"(\" #\\( #\\) ; )\n 1])\n#| ( |#\n(g)"), None);
        assert_eq!(check(""), None);
        assert_eq!(check("(a #\\space) (b)"), None);
    }

    #[test]
    fn finds_the_stray_close_the_mismatch_and_the_unclosed_form_with_positions() {
        assert_eq!(check("(a)\n  (b))"), Some(Problem::ExtraClose { line: 2, col: 6 }));
        assert_eq!(check("(a [b)\n"), Some(Problem::Mismatch { open: ('[', 1, 4), close: (')', 1, 6) }));
        assert_eq!(check("(define (f)\n  (let ([x 1])\n    (g x)\n"), Some(Problem::Unclosed { outermost: (1, 1), innermost: (2, 3), count: 2 }));
    }

    #[test]
    fn explains_chez_reader_errors_in_terms_of_lines() {
        let src = "(register-pass! 'p\n  (lambda (k)\n    k))\n  )\n";
        let msg = "Exception in read: unexpected close parenthesis at char 44 of #<input port string>";
        let e = explain(msg, src);
        assert!(e.contains("[line 4, column 3: `)`]") && e.contains("an extra `)` at line 4, column 3"), "{e}");
        let src = "(define (f x)\n  (+ x 1)\n";
        let e = explain("Exception in read: unexpected end-of-file reading list at char 23 of #<input port string>", src);
        assert!(e.contains("1 parenthesis is never closed") && e.contains("opens at line 1, column 1 (`(define (f x)`)") && !e.contains("[line"), "{e}");
        let e = explain("Exception in read: unexpected end-of-file reading list at char 0 of #<input port string>", "(a\n  (b\n");
        assert!(e.contains("2 parentheses are never closed") && e.contains("the innermost at line 2, column 3"), "{e}");
        // other errors are untouched
        assert_eq!(explain("Exception: variable foo is not bound", "(foo)"), "Exception: variable foo is not bound");
    }
}
