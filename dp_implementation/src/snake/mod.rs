pub mod mdp;

/*
 * Snake game
 *
 * Environment is a width x height blocked board
 * (0, 0) = (Top, Left)
 * Snake and fruit is randomly placed in a block
 * The goal of the snake is to get to the fruit and eat it
 *
 *     0    1;h   2
 *     3    4     5
 *     6    7     8;f
 */

pub const WIDTH: i32 = 5;
pub const HEIGHT: i32 = 5;

// pub fn add(left: u64, right: u64) -> u64 {
//     left + right
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
