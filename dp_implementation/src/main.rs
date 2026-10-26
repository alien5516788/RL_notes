pub mod snake;
pub mod policy_evaluation;
pub mod policy_iteration;
pub mod value_iteration;

fn main() {
    // policy_evaluation::run();
    // policy_iteration::run();
    value_iteration::run()
}
