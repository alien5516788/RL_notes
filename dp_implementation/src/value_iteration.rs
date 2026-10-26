use std::collections::HashMap;

use crate::snake::{HEIGHT, WIDTH};
use crate::snake::mdp::{State, Action, world_model, reward, GAMMA};


// Value (V) - learnt
pub fn v_table(states: &Vec<State>) -> HashMap<State, f32> {
    let mut table: HashMap<State, f32> = HashMap::new();

    for s in states {
        table.insert(s.clone(), 0.0);
    }

    table
}

pub fn value(s: &State, v_table: &HashMap<State, f32>) -> f32 {
    v_table[s]
}


// Value iteration
/*
 * Value is learnt by continuously improving the value of each state
 * by choosing the action with the highest expected return.
 *
 *     Q(s, a) = Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
 *     V(s) <- max_a Q(s, a)
 */
pub fn value_iteration(
    states: &Vec<State>,
    actions: &Vec<Action>,
    v_table: &mut HashMap<State, f32>
) {
    let mut new_table = HashMap::new();

    for s in states {
        // V(s)
        let mut v = f32::NEG_INFINITY; // max_a Q(s, a)

        for a in actions {
            // Q(s, a)
            let mut action_value = 0.0;

            // Σ_s'
            for s_next in states {

                // P(s' | s, a)
                let transition_probability = world_model(s, a, s_next);

                // R(s, a, s')
                let immediate_reward = reward(s, a, s_next);

                // V(s')
                let future_value = value(s_next, v_table);

                // Add the next state contribution to Q(s, a)
                action_value += transition_probability * (immediate_reward + (GAMMA * future_value));
            }

            // V(s) <- max_a Q(s, a)
            if action_value > v {
                v = action_value; // max_a Q(s, a)
            }
        }

        new_table.insert(s.clone(), v);
    }

    *v_table = new_table;
}


pub fn run() {
    let states = State::all();
    let actions = Action::all();
    let mut v_table = v_table(&states);

    println!("\nInitial table");
    print_v_table(&v_table, (3, 0));

    // Train
    for i in 0..50 {
        value_iteration(&states, &actions, &mut v_table);

        println!("\nIteration {}", i + 1);
        print_v_table(&v_table, (3, 0));
    }

    // Test
    todo!()
}


fn print_v_table(v_table: &HashMap<State, f32>, fruit: (i32, i32)) {
    println!("Fruit: x = {} y = {}", fruit.0, fruit.1);

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let state = State {
                head: (x, y),
                fruit,
            };

            let value = v_table.get(&state).unwrap();

            print!("{:7.2}", value);
        }

        println!();
    }
}
