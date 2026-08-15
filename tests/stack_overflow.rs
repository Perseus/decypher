//! Regression tests for expressions that exhausted the legacy AST builder's
//! default test-thread stack.

use decypher::ast::ToCypher;
use decypher::ast::arena::{AstArenas, ExprKind, LogicalOperator, build_expression_arena};
use decypher::parse_cst;
use decypher::syntax::SyntaxKind;
use decypher::syntax::ast::AstNode;
use decypher::syntax::ast::expressions::Expression;
use decypher::{ErrorKind, parse, sema};

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

fn assert_parser_recursion_limit(query: &str) {
    let parsed = parse_cst(query);
    assert!(parsed.errors.iter().any(|error| matches!(
        error.kind,
        ErrorKind::RecursionLimitExceeded {
            phase: "parser",
            ..
        }
    )));
}

fn assert_ast_recursion_limit(query: &str) {
    let cst = parse_cst(query);
    assert!(cst.errors.is_empty(), "CST errors: {:?}", cst.errors);
    let error = parse(query).expect_err("the legacy AST recursion budget should reject the query");
    assert!(matches!(
        error.kind,
        ErrorKind::RecursionLimitExceeded {
            phase: "AST builder",
            ..
        }
    ));
}

/// Truly nested syntax is rejected before the recursive Pratt parser can
/// exhaust the native stack.
#[test]
fn parser_rejects_excessive_expression_nesting_without_overflowing() {
    let depth = 1_000;
    let parentheses = format!("RETURN {}1{}", "(".repeat(depth), ")".repeat(depth));
    let unary = format!("RETURN {}true", "NOT ".repeat(depth));
    let power = format!("RETURN {}", vec!["1"; depth].join(" ^ "));

    assert_parser_recursion_limit(&parentheses);
    assert_parser_recursion_limit(&unary);
    assert_parser_recursion_limit(&power);
}

/// Recursive grammar families outside Pratt expressions share the same
/// bounded counter.
#[test]
fn parser_rejects_excessive_label_and_subquery_nesting_without_overflowing() {
    let labels = format!("MATCH (n:{}Label) RETURN n", "!".repeat(1_000));
    let unions = format!(
        "CALL {{ {} }} RETURN 1",
        vec!["RETURN 1"; 1_000].join(" UNION ")
    );

    assert_parser_recursion_limit(&labels);
    assert_parser_recursion_limit(&unions);
}

/// Flat non-logical CST chains still use the legacy recursive boxed builder.
/// They fail with a typed diagnostic until that representation is migrated.
#[test]
fn ast_builder_rejects_remaining_recursive_chains_without_overflowing() {
    let subtraction = format!("RETURN {}", vec!["1"; 1_000].join(" - "));
    let property = format!(
        "RETURN n{}",
        (0..1_000)
            .map(|index| format!(".p{index}"))
            .collect::<String>()
    );

    assert_ast_recursion_limit(&subtraction);
    assert_ast_recursion_limit(&property);
}

/// Label OR/AND chains use ordered operand vectors in the public AST, so
/// printing and HIR relationship-type lowering stay stack-safe.
#[test]
fn public_pipeline_handles_large_label_expression() {
    let labels = (0..1_000)
        .map(|index| format!("L{index}"))
        .collect::<Vec<_>>()
        .join("|");
    let node_query = format!("MATCH (n:{labels}) RETURN n");
    let parsed = parse(node_query.as_str()).expect("flat label AST construction should succeed");
    sema::analyze(&parsed).expect("semantic analysis should be stack-safe");
    assert_eq!(parsed.to_cypher().matches('|').count(), 999);

    #[cfg(feature = "hir")]
    {
        let relationship_query = format!("MATCH ()-[:{labels}]->() RETURN 1");
        decypher::analyze(relationship_query.as_str())
            .expect("relationship type lowering should be stack-safe");
    }
}
