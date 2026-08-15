//! Regression tests for expressions that exhausted the legacy AST builder's
//! default test-thread stack.

use decypher::ast::arena::{AstArenas, ExprKind, LogicalOperator, build_expression_arena};
use decypher::parse_cst;
use decypher::syntax::SyntaxKind;
use decypher::syntax::ast::AstNode;
use decypher::syntax::ast::expressions::Expression;

const QUERY_PREFIX: &str = "MATCH (n)\nWHERE (";
const QUERY_SUFFIX: &str = ")\nRETURN n";
const LABEL_PREDICATE_COUNT: usize = 250;

fn large_disjunction_query() -> String {
    let predicates = (0..LABEL_PREDICATE_COUNT)
        .map(|index| format!("n:Label{index}"))
        .collect::<Vec<_>>()
        .join(" OR ");

    format!("{QUERY_PREFIX}{predicates}{QUERY_SUFFIX}")
}

/// The lossless CST parser can read the generated disjunction without errors.
#[test]
fn parse_cst_large_label_predicate_disjunction() {
    let query = large_disjunction_query();
    let parsed = parse_cst(query.as_str());

    assert!(
        parsed.errors.is_empty(),
        "parse errors: {:?}",
        parsed.errors
    );
    assert_eq!(
        parsed
            .tree
            .descendants()
            .filter(|node| node.kind() == SyntaxKind::OR_EXPR)
            .count(),
        LABEL_PREDICATE_COUNT - 1
    );
}

/// The iterative arena builder flattens the same long disjunction into one
/// logical node and does not consume one native stack frame per predicate.
#[test]
fn build_large_label_predicate_disjunction_iteratively() {
    let query = large_disjunction_query();
    let parsed = parse_cst(query.as_str());
    let cst_root = parsed
        .tree
        .descendants()
        .filter(|node| node.kind() == SyntaxKind::OR_EXPR)
        .last()
        .and_then(Expression::cast)
        .expect("generated query should contain an OR expression");
    let mut arenas = AstArenas::new();

    let root = build_expression_arena(cst_root, &mut arenas)
        .expect("arena construction should not overflow the stack");

    let ExprKind::Logical { op, operands } = &arenas.expressions.get(root).kind else {
        panic!("expected one flat logical expression");
    };
    assert_eq!(*op, LogicalOperator::Or);
    assert_eq!(operands.len(), LABEL_PREDICATE_COUNT);
}
