use decypher::ast::arena::LogicalOperator;
use decypher::ast::query::Query;
use decypher::ast::visit::Visit;
use decypher::parse;

#[derive(Default)]
struct LogicalOperatorCollector {
    operators: Vec<LogicalOperator>,
}

impl<'ast> Visit<'ast> for LogicalOperatorCollector {
    fn visit_logical_operator(&mut self, operator: &'ast LogicalOperator) {
        self.operators.push(*operator);
    }
}

#[test]
fn visitor_observes_logical_operators() {
    let query: Query = parse("RETURN true OR false AND true").expect("query should parse");
    let mut collector = LogicalOperatorCollector::default();

    collector.visit_query(&query);

    assert_eq!(
        collector.operators,
        vec![LogicalOperator::Or, LogicalOperator::And]
    );
}
