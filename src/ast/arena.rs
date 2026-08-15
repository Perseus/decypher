//! Arena-backed AST nodes.
//!
//! This model stores expression children as typed IDs instead of nesting
//! boxed expressions. It exists alongside the legacy recursive expression
//! model while CST construction and AST consumers migrate in later changes.

use crate::arena::{Arena, ArenaId};
use crate::error::Span;

use super::expr::{ComparisonOperator, StringLiteral, UnaryOperator};
use super::names::{PropertyKeyName, SymbolicName, Variable};
use super::pattern::{Pattern, RelationshipsPattern};
use super::query::RegularQuery;

pub use super::build_arena::build_expression_arena;

/// An ID for an expression in [`AstArenas::expressions`].
pub type ExprId = ArenaId<ExpressionNode>;

/// An ID for a label expression in [`AstArenas::label_expressions`].
pub type LabelExprId = ArenaId<LabelExpressionNode>;

/// Storage owned by one parsed [`super::query::Query`].
///
/// IDs are meaningful only with the `AstArenas` belonging to that query.
#[derive(Debug, Clone, PartialEq)]
pub struct AstArenas {
    /// All expression nodes in the query, including nested subqueries.
    pub expressions: Arena<ExpressionNode>,
    /// All label and relationship-type expression nodes in the query.
    pub label_expressions: Arena<LabelExpressionNode>,
}

impl AstArenas {
    /// Create empty AST storage.
    pub const fn new() -> Self {
        Self {
            expressions: Arena::new(),
            label_expressions: Arena::new(),
        }
    }

    /// Allocate an expression node.
    pub fn alloc_expression(&mut self, kind: ExprKind, span: Span) -> ExprId {
        self.expressions.alloc(ExpressionNode { kind, span })
    }

    /// Allocate a label-expression node.
    pub fn alloc_label_expression(&mut self, kind: LabelExprKind, span: Span) -> LabelExprId {
        self.label_expressions
            .alloc(LabelExpressionNode { kind, span })
    }
}

impl Default for AstArenas {
    fn default() -> Self {
        Self::new()
    }
}

/// One expression node stored in an [`AstArenas`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressionNode {
    /// The expression form and its references to child nodes.
    pub kind: ExprKind,
    /// Byte-offset span of the complete expression.
    pub span: Span,
}

/// An arena-backed Cypher expression.
///
/// Recursive relationships use [`ExprId`]s. Logical `AND` and `OR` use one
/// node with many operands, preventing a long chain from becoming a deep
/// binary tree.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    /// A literal value.
    Literal(LiteralValue),
    /// A variable reference.
    Variable(Variable),
    /// A query parameter reference without the leading `$`.
    Parameter(SymbolicName),
    /// `base.property`.
    PropertyLookup {
        /// Expression whose property is accessed.
        base: ExprId,
        /// Property name.
        property: PropertyKeyName,
    },
    /// `base:Label`.
    NodeLabels {
        /// Expression tested for labels.
        base: ExprId,
        /// Label expressions applied to the base.
        labels: Vec<LabelExprId>,
    },
    /// A non-`AND`/`OR` binary operation.
    Binary {
        /// Operator.
        op: BinaryOperator,
        /// Left operand.
        lhs: ExprId,
        /// Right operand.
        rhs: ExprId,
    },
    /// An associative logical operation stored as one flat operand list.
    Logical {
        /// `AND` or `OR`.
        op: LogicalOperator,
        /// Operands in source order.
        operands: Vec<ExprId>,
    },
    /// A unary operation.
    Unary {
        /// Operator.
        op: UnaryOperator,
        /// Operand.
        operand: ExprId,
    },
    /// A chained comparison such as `a < b <= c`.
    Comparison {
        /// Leftmost operand.
        lhs: ExprId,
        /// Remaining operator and operand pairs.
        operators: Vec<(ComparisonOperator, ExprId)>,
    },
    /// `list[index]`.
    ListIndex {
        /// List expression.
        list: ExprId,
        /// Index expression.
        index: ExprId,
    },
    /// `list[start..end]`.
    ListSlice {
        /// List expression.
        list: ExprId,
        /// Optional inclusive start.
        start: Option<ExprId>,
        /// Optional exclusive end.
        end: Option<ExprId>,
    },
    /// `lhs IN rhs`.
    In {
        /// Candidate value.
        lhs: ExprId,
        /// Collection expression.
        rhs: ExprId,
    },
    /// `operand IS NULL` or `operand IS NOT NULL`.
    IsNull {
        /// Tested expression.
        operand: ExprId,
        /// Whether `NOT` is present.
        negated: bool,
    },
    /// A function invocation.
    FunctionCall(FunctionCall),
    /// `count(*)`.
    CountStar,
    /// A `CASE` expression.
    Case(CaseExpression),
    /// A list comprehension.
    ListComprehension(ListComprehension),
    /// A pattern comprehension.
    PatternComprehension(PatternComprehension),
    /// `ALL`, `ANY`, `NONE`, or `SINGLE` over a collection.
    CollectionPredicate(FilterExpression),
    /// An explicitly parenthesized expression.
    Parenthesized(ExprId),
    /// A relationships pattern used as an expression.
    Pattern(RelationshipsPattern),
    /// `EXISTS { ... }`.
    Exists(ExistsInner),
    /// `COUNT { ... }`.
    CountSubquery(Box<RegularQuery>),
    /// `COLLECT { ... }`.
    CollectSubquery(Box<RegularQuery>),
    /// A map projection.
    MapProjection(MapProjection),
}

/// Binary operators that are not flattened into [`ExprKind::Logical`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    /// `+`.
    Add,
    /// `-`.
    Subtract,
    /// `*`.
    Multiply,
    /// `/`.
    Divide,
    /// `%`.
    Modulo,
    /// `^`.
    Power,
    /// `XOR`.
    Xor,
}

/// Logical operators represented by a flat list of operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOperator {
    /// `AND`.
    And,
    /// `OR`.
    Or,
}

/// A literal whose nested expressions are arena IDs.
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    /// A signed integer.
    Integer(i64),
    /// An IEEE 754 floating-point number.
    Float(f64),
    /// A decoded string and its optional source spelling.
    String(StringLiteral),
    /// A boolean.
    Boolean(bool),
    /// `null`.
    Null,
    /// A list literal.
    List(Vec<ExprId>),
    /// A map literal in source order.
    Map(Vec<(PropertyKeyName, ExprId)>),
}

/// An arena-backed function call.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    /// Qualified name segments.
    pub name: Vec<SymbolicName>,
    /// Whether the arguments are distinct.
    pub distinct: bool,
    /// Positional arguments.
    pub arguments: Vec<ExprId>,
}

/// An arena-backed `CASE` expression.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseExpression {
    /// Optional generic-case scrutinee.
    pub scrutinee: Option<ExprId>,
    /// `WHEN` and `THEN` expression pairs.
    pub alternatives: Vec<(ExprId, ExprId)>,
    /// Optional `ELSE` expression.
    pub default: Option<ExprId>,
}

/// An arena-backed list comprehension.
#[derive(Debug, Clone, PartialEq)]
pub struct ListComprehension {
    /// Element variable.
    pub variable: Variable,
    /// Source collection.
    pub collection: ExprId,
    /// Optional `WHERE` predicate.
    pub filter: Option<ExprId>,
    /// Optional mapped value.
    pub map: Option<ExprId>,
}

/// An arena-backed pattern comprehension.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternComprehension {
    /// Optional path variable.
    pub variable: Option<Variable>,
    /// Path pattern.
    pub pattern: RelationshipsPattern,
    /// Optional `WHERE` predicate.
    pub where_clause: Option<ExprId>,
    /// Mapped value.
    pub map: ExprId,
}

/// A collection predicate and its arena-backed children.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterExpression {
    /// Predicate kind.
    pub quantifier: CollectionQuantifier,
    /// Element variable.
    pub variable: Variable,
    /// Source collection.
    pub collection: ExprId,
    /// Optional `WHERE` predicate.
    pub predicate: Option<ExprId>,
}

/// Collection predicate kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionQuantifier {
    /// `ALL`.
    All,
    /// `ANY`.
    Any,
    /// `NONE`.
    None,
    /// `SINGLE`.
    Single,
}

/// Inner content of an `EXISTS` expression.
#[derive(Debug, Clone, PartialEq)]
pub enum ExistsInner {
    /// A pattern with an optional predicate.
    Pattern(Pattern, Option<ExprId>),
    /// A full regular query.
    RegularQuery(Box<RegularQuery>),
}

/// An arena-backed map projection.
#[derive(Debug, Clone, PartialEq)]
pub struct MapProjection {
    /// Existing entity or map variable.
    pub base: Variable,
    /// Projection items in source order.
    pub items: Vec<MapProjectionItem>,
}

/// One map-projection item.
#[derive(Debug, Clone, PartialEq)]
pub enum MapProjectionItem {
    /// `.*`.
    AllProperties,
    /// `.property`.
    PropertyLookup(PropertyKeyName),
    /// `key: value`.
    Literal(PropertyKeyName, ExprId),
}

/// One label-expression node stored in an [`AstArenas`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelExpressionNode {
    /// Label-expression form.
    pub kind: LabelExprKind,
    /// Byte-offset span of the complete expression.
    pub span: Span,
}

/// An arena-backed label or relationship-type expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LabelExprKind {
    /// A static label or type name.
    Static(SymbolicName),
    /// A dynamic expression such as `$(expr)`.
    Dynamic(ExprId),
    /// A flat disjunction in source order.
    Or(Vec<LabelExprId>),
    /// A flat conjunction in source order.
    And(Vec<LabelExprId>),
    /// A negated label expression.
    Not(LabelExprId),
    /// An explicit parenthesized group.
    Group(LabelExprId),
}

#[cfg(test)]
mod tests {
    use super::{AstArenas, ExprKind, LogicalOperator};
    use crate::ast::names::{SymbolicName, Variable};
    use crate::error::Span;

    #[test]
    fn stores_large_or_as_one_logical_node() {
        let mut arenas = AstArenas::new();
        let operands = (0..10_000)
            .map(|index| {
                let name = SymbolicName {
                    name: format!("value_{index}"),
                    span: Span::new(0, 0),
                };
                arenas.alloc_expression(ExprKind::Variable(Variable { name }), Span::new(0, 0))
            })
            .collect();

        let root = arenas.alloc_expression(
            ExprKind::Logical {
                op: LogicalOperator::Or,
                operands,
            },
            Span::new(0, 0),
        );

        let ExprKind::Logical { operands, .. } = &arenas.expressions.get(root).kind else {
            panic!("expected logical expression");
        };
        assert_eq!(operands.len(), 10_000);
        assert_eq!(arenas.expressions.len(), 10_001);
    }
}
