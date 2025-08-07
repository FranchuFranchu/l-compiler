use crate::{AnnotatedVar, Captures, Command, Expression, Type};
use serde_derive::{Deserialize, Serialize};

impl Command {
    pub fn from_lexpr_with(name: &str, args: &[&lexpr::Value], command: Command) -> Option<Self> {
        match name {
            "receive" => Some(Self::Receive(
                args[0].as_str().unwrap().to_owned(),
                args[1].as_str().unwrap().to_owned(),
                Box::new(command),
            )),
            "continue" => Some(Self::Continue(
                args[0].as_str().unwrap().to_owned(),
                Box::new(command),
            )),
            "erase" => Some(Self::Erase(
                args[0].as_str().unwrap().to_owned(),
                Box::new(command),
            )),
            "unbox" => Some(Self::Unbox(
                args[0].as_str().unwrap().to_owned(),
                Box::new(command),
            )),
            "duplicate" => Some(Self::Duplicate(
                args[0].as_str().unwrap().to_owned(),
                args[1].as_str().unwrap().to_owned(),
                Box::new(command),
            )),
            _ => None,
        }
    }
    pub fn from_lexpr(value: &lexpr::Value) -> Option<Self> {
        let vec = value.to_ref_vec().unwrap();
        let name = vec[0].as_symbol().unwrap();
        println!("from lexpr {vec:#?}");
        match name {
            "match" => Some(Self::Match(
                vec[1].as_str().unwrap().to_owned(),
                Box::new(Command::from_lexpr(&vec[2])?),
                Box::new(Command::from_lexpr(&vec[3])?),
            )),
            "do" => vec[1..].iter().rfold(Some(None), |acc, i| {
                if let Some(Some(acc)) = acc {
                    let vec = i.to_ref_vec().unwrap();
                    let name = vec[0].as_symbol().unwrap();
                    Some(Some(Command::from_lexpr_with(name, &vec[1..], acc)?))
                } else if let Some(None) = acc {
                    Some(Some(Command::from_lexpr(i)?))
                } else {
                    None
                }
            })?,
            "cut" => Some(Self::Cut(
                Expression::from_lexpr(&vec[1])?,
                Expression::from_lexpr(&vec[2])?,
            )),
            name => Self::from_lexpr_with(
                name,
                &vec[1..vec.len() - 1],
                Command::from_lexpr(&vec.last().unwrap())?,
            ),
        }
    }
}

impl Expression {
    pub fn from_lexpr(value: &lexpr::Value) -> Option<Self> {
        if let Some(name) = value.as_symbol() {
            if name == "unit" {
                return Some(Expression::Unit);
            } else {
                return None;
            }
        }
        let vec = value.to_ref_vec().unwrap();
        let name = vec[0].as_symbol().unwrap();
        match name {
            "var" => Some(Self::Var(vec[1].as_str().unwrap().to_owned())),
            "pair" => Some(Self::Pair(
                Box::new(Expression::from_lexpr(&vec[1])?),
                Box::new(Expression::from_lexpr(&vec[2])?),
            )),
            "left" => Some(Self::Left(
                Box::new(Expression::from_lexpr(&vec[1])?),
                (Type::from_lexpr(&vec[2])?),
            )),
            "right" => Some(Self::Right(
                Box::new(Expression::from_lexpr(&vec[1])?),
                (Type::from_lexpr(&vec[2])?),
            )),
            "chan" => Some(Self::Chan(
                Captures::from_lexpr(&vec[1])?,
                AnnotatedVar::from_lexpr(&vec[2])?,
                Box::new(Command::from_lexpr(&vec[3])?),
            )),
            "box" => Some(Self::Box(
                Captures::from_lexpr(&vec[1])?,
                Box::new(Expression::from_lexpr(&vec[2])?),
            )),
            _ => None,
        }
    }
}

impl Type {
    fn from_lexpr(value: &lexpr::Value) -> Option<Self> {
        if let Some(name) = value.as_symbol() {
            return match name {
                "unit" => Some(Self::Unit),
                "zero" => Some(Self::Zero),
                _ => None,
            };
        }
        let vec = value.to_ref_vec().unwrap();
        let name = vec[0].as_symbol().unwrap();

        match name {
            "pair" => Some(Self::Pair(
                Box::new(Type::from_lexpr(&vec[1])?),
                Box::new(Type::from_lexpr(&vec[2])?),
            )),
            "either" => Some(Self::Either(
                Box::new(Type::from_lexpr(&vec[1])?),
                Box::new(Type::from_lexpr(&vec[2])?),
            )),
            "box" => Some(Self::Box(Box::new(Type::from_lexpr(&vec[1])?))),
            "dual" => Some(Self::Dual(Box::new(Type::from_lexpr(&vec[1])?))),
            _ => None,
        }
    }
}

impl AnnotatedVar {
    fn from_lexpr(value: &lexpr::Value) -> Option<Self> {
        let vec = value.to_ref_vec().unwrap();
        Some(AnnotatedVar {
            name: vec[0].as_str().unwrap().to_owned(),
            r#type: Type::from_lexpr(&vec[1])?,
        })
    }
}

impl Captures {
    fn from_lexpr(value: &lexpr::Value) -> Option<Self> {
        let vec = value.to_ref_vec().unwrap();
        Some(Self(
            vec.iter().map(|x| x.as_str().unwrap().to_owned()).collect(),
        ))
    }
}
