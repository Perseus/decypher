//! Regression tests for expressions that exhausted the legacy AST builder's
//! default test-thread stack.

use decypher::ast::ToCypher;
use decypher::ast::arena::{AstArenas, ExprKind, LogicalOperator, build_expression_arena};
use decypher::ast::expr::ComparisonOperator;
use decypher::parse_cst;
use decypher::syntax::SyntaxKind;
use decypher::syntax::ast::AstNode;
use decypher::syntax::ast::expressions::Expression;
use decypher::{parse, sema};

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
    let span = arenas.expressions.get(root).span;
    assert_eq!(
        &query[span.start..span.end],
        query
            .strip_prefix(QUERY_PREFIX)
            .and_then(|query| query.strip_suffix(QUERY_SUFFIX))
            .expect("generated query should contain its predicate")
    );
}

/// The iterative builder stores one comparison node for a complete comparison
/// chain, preserving the leftmost operand and every operator in source order.
#[test]
fn build_comparison_chain_as_one_arena_node() {
    let query = "RETURN a < b <= c";
    let parsed = parse_cst(query);
    let cst_root = parsed
        .tree
        .descendants()
        .filter(|node| node.kind() == SyntaxKind::COMPARISON_EXPR)
        .last()
        .and_then(Expression::cast)
        .expect("query should contain a comparison expression");
    let mut arenas = AstArenas::new();

    let root = build_expression_arena(cst_root, &mut arenas)
        .expect("comparison construction should succeed");
    let node = arenas.expressions.get(root);
    let ExprKind::Comparison { lhs, operators } = &node.kind else {
        panic!("expected one comparison expression");
    };

    let ExprKind::Variable(lhs) = &arenas.expressions.get(*lhs).kind else {
        panic!("expected the leftmost operand to be a variable");
    };
    assert_eq!(lhs.name.name, "a");
    assert_eq!(operators.len(), 2);
    assert_eq!(operators[0].0, ComparisonOperator::Lt);
    assert_eq!(operators[1].0, ComparisonOperator::Le);
    assert_eq!(&query[node.span.start..node.span.end], "a < b <= c");
}

/// The public AST, semantic analysis, printer, and HIR lowering all consume
/// the flat logical representation without rebuilding a left-deep OR tree.
#[test]
fn public_pipeline_handles_large_label_predicate_disjunction() {
    let query = large_disjunction_query();
    let parsed = parse(query.as_str()).expect("public parsing should be stack-safe");

    sema::analyze(&parsed).expect("semantic analysis should be stack-safe");
    assert_eq!(
        parsed.to_cypher().matches(" OR ").count(),
        LABEL_PREDICATE_COUNT - 1
    );

    #[cfg(feature = "hir")]
    {
        let hir = decypher::analyze(parsed).expect("HIR lowering should be stack-safe");
        assert!(hir.arenas.expressions.iter().any(|(_, expression)| {
            matches!(
                &expression.kind,
                decypher::hir::ExprKind::Logical { operands, .. }
                    if operands.len() == LABEL_PREDICATE_COUNT
            )
        }));
    }
}
