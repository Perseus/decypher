//! Iterative CST to arena-backed AST construction.

use crate::ast::arena::{
    AstArenas, BinaryOperator, ExprId, ExprKind, LabelExprId, LabelExprKind, LogicalOperator,
};
use crate::ast::expr::{ComparisonOperator, UnaryOperator};
use crate::ast::names::{SymbolicName, Variable};
use crate::error::{CypherError, ErrorKind, Result, Span};
use crate::syntax::ast::expressions::{Atom, BinOp, BinaryExpr, Expression, UnOp, UnaryExpr};
use crate::syntax::ast::patterns::{
    LabelExprNode as CstLabelExprNode, LabelExpression as CstLabelExpression, NodeLabels,
};
use crate::syntax::SyntaxNode;
use crate::syntax::ast::AstNode;

enum ExpressionTask {
    Visit(Expression),
    FinishBinary {
        op: BinOp,
        span: Span,
    },
    FinishLogical {
        op: LogicalOperator,
        operand_count: usize,
        span: Span,
    },
    FinishUnary {
        op: UnaryOperator,
        span: Span,
    },
    FinishParenthesized {
        span: Span,
    },
    FinishLabels {
        labels: Vec<CstLabelExpression>,
        span: Span,
    },
}

enum LabelTask {
    Visit(CstLabelExprNode),
    FinishOr { count: usize, span: Span },
    FinishAnd { count: usize, span: Span },
    FinishNot { span: Span },
    FinishGroup { span: Span },
}

/// Build one CST expression into `arenas` without recursive expression calls.
///
/// This first migration step supports variables, unary expressions,
/// parenthesized expressions, binary operators, and static label predicates.
/// Other atom forms will be added before the arena-backed AST becomes the
/// default public representation.
pub fn build_expression_arena(expression: Expression, arenas: &mut AstArenas) -> Result<ExprId> {
    let mut tasks = vec![ExpressionTask::Visit(expression)];
    let mut results = Vec::new();

    while let Some(task) = tasks.pop() {
        match task {
            ExpressionTask::Visit(expression) => match expression {
                Expression::BinaryExpr(binary) => schedule_binary(binary, &mut tasks)?,
                Expression::UnaryExpr(unary) => schedule_unary(unary, &mut tasks)?,
                Expression::Atom(Atom::Variable(variable)) => {
                    let span = span_of(variable.syntax());
                    let name = variable
                        .name()
                        .ok_or_else(|| internal("missing variable name", span))?;
                    let name = SymbolicName {
                        name: symbolic_name_text(&name),
                        span: span_of(name.syntax()),
                    };
                    results
                        .push(arenas.alloc_expression(ExprKind::Variable(Variable { name }), span));
                }
                Expression::Atom(Atom::Parenthesized(parenthesized)) => {
                    let span = span_of(parenthesized.syntax());
                    let inner = parenthesized
                        .expr()
                        .ok_or_else(|| internal("missing parenthesized expression", span))?;
                    tasks.push(ExpressionTask::FinishParenthesized { span });
                    tasks.push(ExpressionTask::Visit(inner));
                }
                Expression::Atom(atom) => {
                    return Err(internal(
                        "arena expression builder does not support this atom yet",
                        span_of(atom.syntax()),
                    ));
                }
            },
            ExpressionTask::FinishBinary { op, span } => {
                let rhs = results
                    .pop()
                    .ok_or_else(|| internal("missing binary rhs result", span))?;
                let lhs = results
                    .pop()
                    .ok_or_else(|| internal("missing binary lhs result", span))?;
                let kind = finish_binary(op, lhs, rhs, span)?;
                results.push(arenas.alloc_expression(kind, span));
            }
            ExpressionTask::FinishLogical {
                op,
                operand_count,
                span,
            } => {
                let operands = take_results(&mut results, operand_count, span)?;
                results.push(arenas.alloc_expression(ExprKind::Logical { op, operands }, span));
            }
            ExpressionTask::FinishUnary { op, span } => {
                let operand = results
                    .pop()
                    .ok_or_else(|| internal("missing unary operand result", span))?;
                results.push(arenas.alloc_expression(ExprKind::Unary { op, operand }, span));
            }
            ExpressionTask::FinishParenthesized { span } => {
                let inner = results
                    .pop()
                    .ok_or_else(|| internal("missing parenthesized result", span))?;
                results.push(arenas.alloc_expression(ExprKind::Parenthesized(inner), span));
            }
            ExpressionTask::FinishLabels { labels, span } => {
                let base = results
                    .pop()
                    .ok_or_else(|| internal("missing label predicate base", span))?;
                let labels = labels
                    .into_iter()
                    .map(|label| build_label_expression_arena(label, arenas))
                    .collect::<Result<Vec<_>>>()?;
                results.push(arenas.alloc_expression(ExprKind::NodeLabels { base, labels }, span));
            }
        }
    }

    match results.as_slice() {
        [result] => Ok(*result),
        _ => Err(internal(
            "expression work stack produced an invalid result count",
            Span::new(0, 0),
        )),
    }
}

fn schedule_binary(binary: BinaryExpr, tasks: &mut Vec<ExpressionTask>) -> Result<()> {
    let span = span_of(binary.syntax());
    let op = binary
        .op_kind()
        .ok_or_else(|| internal("unknown binary operator", span))?;

    if let Some(logical_op) = logical_operator(op) {
        let operands = collect_logical_operands(binary, op)?;
        tasks.push(ExpressionTask::FinishLogical {
            op: logical_op,
            operand_count: operands.len(),
            span,
        });
        tasks.extend(operands.into_iter().rev().map(ExpressionTask::Visit));
        return Ok(());
    }

    let lhs = binary
        .lhs()
        .ok_or_else(|| internal("missing binary lhs", span))?;

    if op == BinOp::HasLabel {
        let labels = binary
            .syntax()
            .children()
            .filter_map(NodeLabels::cast)
            .filter_map(|labels| labels.expression())
            .collect();
        tasks.push(ExpressionTask::FinishLabels { labels, span });
        tasks.push(ExpressionTask::Visit(lhs));
        return Ok(());
    }

    if matches!(op, BinOp::IsNull | BinOp::IsNotNull) {
        return Err(internal(
            "null predicates are not supported by the arena builder yet",
            span,
        ));
    }

    let rhs = binary
        .rhs()
        .ok_or_else(|| internal("missing binary rhs", span))?;
    tasks.push(ExpressionTask::FinishBinary { op, span });
    tasks.push(ExpressionTask::Visit(rhs));
    tasks.push(ExpressionTask::Visit(lhs));
    Ok(())
}

fn schedule_unary(unary: UnaryExpr, tasks: &mut Vec<ExpressionTask>) -> Result<()> {
    let span = span_of(unary.syntax());
    let op = match unary
        .op()
        .ok_or_else(|| internal("unknown unary operator", span))?
    {
        UnOp::Not => UnaryOperator::Not,
        UnOp::Neg => UnaryOperator::Negate,
        UnOp::Pos => UnaryOperator::Plus,
    };
    let operand = unary
        .operand()
        .ok_or_else(|| internal("missing unary operand", span))?;
    tasks.push(ExpressionTask::FinishUnary { op, span });
    tasks.push(ExpressionTask::Visit(operand));
    Ok(())
}

fn collect_logical_operands(binary: BinaryExpr, op: BinOp) -> Result<Vec<Expression>> {
    let span = span_of(binary.syntax());
    let mut operands_reversed = Vec::new();
    let mut current = Expression::BinaryExpr(binary);

    loop {
        let Expression::BinaryExpr(binary) = current else {
            operands_reversed.push(current);
            break;
        };
        if binary.op_kind() != Some(op) {
            operands_reversed.push(Expression::BinaryExpr(binary));
            break;
        }
        operands_reversed.push(
            binary
                .rhs()
                .ok_or_else(|| internal("missing logical rhs", span))?,
        );
        current = binary
            .lhs()
            .ok_or_else(|| internal("missing logical lhs", span))?;
    }

    operands_reversed.reverse();
    Ok(operands_reversed)
}

fn logical_operator(op: BinOp) -> Option<LogicalOperator> {
    match op {
        BinOp::And => Some(LogicalOperator::And),
        BinOp::Or => Some(LogicalOperator::Or),
        _ => None,
    }
}

fn finish_binary(op: BinOp, lhs: ExprId, rhs: ExprId, span: Span) -> Result<ExprKind> {
    let kind = match op {
        BinOp::Xor => ExprKind::Binary {
            op: BinaryOperator::Xor,
            lhs,
            rhs,
        },
        BinOp::Add => ExprKind::Binary {
            op: BinaryOperator::Add,
            lhs,
            rhs,
        },
        BinOp::Sub => ExprKind::Binary {
            op: BinaryOperator::Subtract,
            lhs,
            rhs,
        },
        BinOp::Mul => ExprKind::Binary {
            op: BinaryOperator::Multiply,
            lhs,
            rhs,
        },
        BinOp::Div => ExprKind::Binary {
            op: BinaryOperator::Divide,
            lhs,
            rhs,
        },
        BinOp::Mod => ExprKind::Binary {
            op: BinaryOperator::Modulo,
            lhs,
            rhs,
        },
        BinOp::Power => ExprKind::Binary {
            op: BinaryOperator::Power,
            lhs,
            rhs,
        },
        BinOp::Eq => comparison(lhs, ComparisonOperator::Eq, rhs),
        BinOp::Ne => comparison(lhs, ComparisonOperator::Ne, rhs),
        BinOp::Lt => comparison(lhs, ComparisonOperator::Lt, rhs),
        BinOp::Gt => comparison(lhs, ComparisonOperator::Gt, rhs),
        BinOp::Le => comparison(lhs, ComparisonOperator::Le, rhs),
        BinOp::Ge => comparison(lhs, ComparisonOperator::Ge, rhs),
        BinOp::RegexMatch => comparison(lhs, ComparisonOperator::RegexMatch, rhs),
        BinOp::StartsWith => comparison(lhs, ComparisonOperator::StartsWith, rhs),
        BinOp::EndsWith => comparison(lhs, ComparisonOperator::EndsWith, rhs),
        BinOp::Contains => comparison(lhs, ComparisonOperator::Contains, rhs),
        BinOp::In => ExprKind::In { lhs, rhs },
        unsupported => {
            return Err(internal(
                &format!("unsupported arena binary operator: {unsupported:?}"),
                span,
            ));
        }
    };
    Ok(kind)
}

fn comparison(lhs: ExprId, op: ComparisonOperator, rhs: ExprId) -> ExprKind {
    ExprKind::Comparison {
        lhs,
        operators: vec![(op, rhs)],
    }
}

fn build_label_expression_arena(
    expression: CstLabelExpression,
    arenas: &mut AstArenas,
) -> Result<LabelExprId> {
    let span = span_of(expression.syntax());
    let root = expression
        .root()
        .ok_or_else(|| internal("missing label expression root", span))?;
    let mut tasks = vec![LabelTask::Visit(root)];
    let mut results = Vec::new();

    while let Some(task) = tasks.pop() {
        match task {
            LabelTask::Visit(node) => {
                let span = span_of(node.syntax());
                match node {
                    CstLabelExprNode::Or(or) => {
                        let items = or.items().collect::<Vec<_>>();
                        tasks.push(LabelTask::FinishOr {
                            count: items.len(),
                            span,
                        });
                        tasks.extend(items.into_iter().rev().map(LabelTask::Visit));
                    }
                    CstLabelExprNode::And(and) => {
                        let items = and.items().collect::<Vec<_>>();
                        tasks.push(LabelTask::FinishAnd {
                            count: items.len(),
                            span,
                        });
                        tasks.extend(items.into_iter().rev().map(LabelTask::Visit));
                    }
                    CstLabelExprNode::Not(not) => {
                        let inner = not
                            .inner()
                            .ok_or_else(|| internal("missing inner label expression", span))?;
                        tasks.push(LabelTask::FinishNot { span });
                        tasks.push(LabelTask::Visit(inner));
                    }
                    CstLabelExprNode::Paren(parenthesized) => {
                        let inner = parenthesized
                            .inner()
                            .ok_or_else(|| internal("missing grouped label expression", span))?;
                        tasks.push(LabelTask::FinishGroup { span });
                        tasks.push(LabelTask::Visit(inner));
                    }
                    CstLabelExprNode::Atom(atom) => {
                        let symbolic_name = if let Some(label) = atom.node_label() {
                            label.name().and_then(|name| name.symbolic_name())
                        } else if let Some(rel_type) = atom.rel_type_name() {
                            rel_type.symbolic_name()
                        } else {
                            return Err(internal(
                                "dynamic label expressions are not supported yet",
                                span,
                            ));
                        }
                        .ok_or_else(|| internal("missing static label name", span))?;
                        let name = SymbolicName {
                            name: symbolic_name_text(&symbolic_name),
                            span: span_of(symbolic_name.syntax()),
                        };
                        results
                            .push(arenas.alloc_label_expression(LabelExprKind::Static(name), span));
                    }
                }
            }
            LabelTask::FinishOr { count, span } => {
                let items = take_results(&mut results, count, span)?;
                results.push(arenas.alloc_label_expression(LabelExprKind::Or(items), span));
            }
            LabelTask::FinishAnd { count, span } => {
                let items = take_results(&mut results, count, span)?;
                results.push(arenas.alloc_label_expression(LabelExprKind::And(items), span));
            }
            LabelTask::FinishNot { span } => {
                let inner = results
                    .pop()
                    .ok_or_else(|| internal("missing negated label result", span))?;
                results.push(arenas.alloc_label_expression(LabelExprKind::Not(inner), span));
            }
            LabelTask::FinishGroup { span } => {
                let inner = results
                    .pop()
                    .ok_or_else(|| internal("missing grouped label result", span))?;
                results.push(arenas.alloc_label_expression(LabelExprKind::Group(inner), span));
            }
        }
    }

    match results.as_slice() {
        [result] => Ok(*result),
        _ => Err(internal("invalid label expression result count", span)),
    }
}

fn take_results<T: Copy>(results: &mut Vec<T>, count: usize, span: Span) -> Result<Vec<T>> {
    let start = results
        .len()
        .checked_sub(count)
        .ok_or_else(|| internal("work stack result underflow", span))?;
    Ok(results.drain(start..).collect())
}

fn symbolic_name_text(symbolic_name: &crate::syntax::ast::top_level::SymbolicName) -> String {
    symbolic_name
        .ident_token()
        .map(|token| token.unescape())
        .unwrap_or_else(|| symbolic_name.syntax().text().to_string())
}

fn span_of(node: &SyntaxNode) -> Span {
    let range = node.text_range();
    Span::new(range.start().into(), range.end().into())
}

fn internal(message: &str, span: Span) -> CypherError {
    CypherError {
        kind: ErrorKind::Internal {
            message: message.to_string(),
        },
        span,
        source_label: None,
        notes: Vec::new(),
        source: None,
    }
}
