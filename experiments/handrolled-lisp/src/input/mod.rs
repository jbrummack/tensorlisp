use crate::{
    stupid_lisp::{Builtin, Input, Leaf},
    input::InputError::{UnexpectedChar, UnexpectedEOF},
};
use lasso::{Rodeo, Spur};
use std::str::Chars;

pub struct ListInput<'a> {
    chars: Chars<'a>,
    line: u32,
    col: u32,
    scratchpad: String,
    pub interner: Rodeo,
    pub strings: Rodeo,
    is_escaped: bool,
}

/*#[derive(Debug, Clone)]
pub enum List {
    Bool(bool),
    Int(i64),
    Float(f32),
    String(String),
    Keyword(Spur),
    List(Vec<Self>),
    Nil,
}*/
pub struct Span {
    pub start: (u32, u32),
    pub end: (u32, u32),
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum InputError {
    #[error("Unexpected EOF")]
    UnexpectedEOF,
    #[error("Unexpected {what:?} at line{line}:{col}")]
    UnexpectedChar { what: char, line: u32, col: u32 },
}

impl<'a> ListInput<'a> {
    pub fn resolve_str(&self, kw: Spur) -> &str {
        self.interner.resolve(&kw)
    }
    pub fn new(data: &'a str) -> std::io::Result<Self> {
        Ok(Self {
            chars: data.chars(),
            line: 0,
            col: 0,
            scratchpad: String::with_capacity(64),
            interner: Rodeo::new(),
            is_escaped: false,
            strings: Rodeo::new(),
        })
    }

    /// Parse a single top-level expression (atom, string, or list).
    pub fn parse(&mut self) -> Result<Input, InputError> {
        self.skip_whitespace();
        match self.peek() {
            Some('(') => {
                self.advance();
                self.parse_list()
            }
            Some('"') => {
                self.advance();
                self.parse_string().map(Input::Leaf)
            }
            Some(')') => Err(UnexpectedChar {
                what: ')',
                line: self.line,
                col: self.col,
            }),
            Some(_) => self.parse_atom().map(Input::Leaf),
            None => Err(UnexpectedEOF),
        }
    }
    pub fn spurdump(&self) {
        for (k, v) in self.interner.iter() {
            println!("{v}:{k:?}")
        }
    }
    /// Parse every top-level expression until EOF (a whole file/program).
    pub fn parse_all(&mut self) -> Result<Vec<Input>, InputError> {
        let mut out = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek().is_none() {
                break;
            }
            out.push(self.parse()?);
        }
        Ok(out)
    }

    /// Called right after the opening '(' has been consumed.
    pub fn parse_list(&mut self) -> Result<Input, InputError> {
        let mut items = Vec::new();
        loop {
            self.skip_whitespace();
            match self.peek() {
                Some(')') => {
                    self.advance();
                    return Ok(Input::List(items));
                }
                Some(_) => items.push(self.parse()?),
                None => return Err(UnexpectedEOF),
            }
        }
    }

    /// Called right after the opening '"' has been consumed.
    pub fn parse_string(&mut self) -> Result<Leaf, InputError> {
        let should_escape = self.is_escaped;
        self.is_escaped = false;
        match self.advance() {
            Some('\\') => {
                if should_escape {
                    self.scratchpad.push('\\');
                } else {
                    self.is_escaped = true;
                }
                self.parse_string()
            }
            Some('"') => {
                if should_escape {
                    self.scratchpad.push('"');
                    self.parse_string()
                } else {
                    let res = Leaf::String(self.strings.get_or_intern(&self.scratchpad));
                    self.scratchpad.clear();
                    Ok(res)
                }
            }
            Some(ch) if should_escape => {
                // handle common escape sequences, fall back to literal char
                match ch {
                    'n' => self.scratchpad.push('\n'),
                    't' => self.scratchpad.push('\t'),
                    'r' => self.scratchpad.push('\r'),
                    '0' => self.scratchpad.push('\0'),
                    other => self.scratchpad.push(other),
                }
                self.parse_string()
            }
            Some(ch) => {
                self.scratchpad.push(ch);
                self.parse_string()
            }
            None => Err(UnexpectedEOF),
        }
    }

    /// Parse a bare token: nil / true / false / int / float / keyword(symbol).
    fn parse_atom(&mut self) -> Result<Leaf, InputError> {
        self.scratchpad.clear();
        while let Some(c) = self.peek() {
            if c.is_whitespace() || c == '(' || c == ')' || c == '"' {
                break;
            }
            self.scratchpad.push(c);
            self.advance();
        }

        if self.scratchpad.is_empty() {
            return match self.peek() {
                Some(c) => Err(UnexpectedChar {
                    what: c,
                    line: self.line,
                    col: self.col,
                }),
                None => Err(UnexpectedEOF),
            };
        }

        let token = std::mem::take(&mut self.scratchpad);
        match token.as_str() {
            "nil" => Ok(Leaf::Nil),
            "true" => Ok(Leaf::Bool(true)),
            "false" => Ok(Leaf::Bool(false)),
            _ => {
                if let Ok(i) = token.parse::<i32>() {
                    Ok(Leaf::Int(i))
                } else if let Ok(f) = token.parse::<f32>() {
                    Ok(Leaf::Float(f.to_ne_bytes()))
                } else {
                    //let spur = self.interner.get_or_intern(&token);
                    Ok(self.build_keyword(&token))
                }
            }
        }
    }
    fn build_keyword(&mut self, token: &str) -> Leaf {
        use Builtin::*;
        Leaf::Lambda(match token {
            "let" => Let,
            "fn" => Fn,
            "+" => Add,
            "-" => Sub,
            "*" => Mul,
            "/" => Div,
            "%" => Mod,
            other => return Leaf::Keyword(self.interner.get_or_intern(other)),
        })
    }
    fn skip_whitespace(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.advance();
                }
                Some(';') => {
                    // line comment
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.advance();
                    }
                }
                _ => break,
            }
        }
    }

    /// Look at the next char without consuming it.
    pub fn peek(&self) -> Option<char> {
        self.chars.clone().next()
    }

    pub fn advance(&mut self) -> Option<char> {
        let next = self.chars.next()?;
        if next == '\n' {
            self.line += 1;
            self.col = 0;
        } else {
            self.col += 1;
        }
        Some(next)
    }
}
