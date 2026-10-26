use std::collections::HashMap;

use crate::snake::{HEIGHT, WIDTH};
use crate::snake::mdp::{State, Action, world_model, reward, GAMMA};


// Policy (π) - defined
fn policy(_s: &State, a: &Action) -> f32 {
    match a {
        Action::Up => 0.15,
        Action::Down => 0.15,
        Action::Left => 0.15,
        Action::Right => 0.15,
        Action::Eat => 0.4,
    }
}


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


/*
 * Value is learnet by continuously interacting with the world_model
 *
 *     V(s) <- Σ_a π(a | s) Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
 */
pub fn policy_evaluation(
    states: &Vec<State>,
    actions: &Vec<Action>,
    v_table: &mut HashMap<State, f32>
) {
    let mut new_table = HashMap::new();

    for s in states {
        // V(s)
        let mut v = 0.0;

        // Σ_a
        for a in actions {
            // π(a | s)
            let action_probability = policy(s, a);

            // Σ_s'
            for s_next in states {
                // P(s' | s, a)
                let transition_probability = world_model(s, a, s_next);

                // R(s, a, s')
                let immediate_reward = reward(s, a, s_next);

                // V(s')
                let future_value = v_table
                    .get(s_next)
                    .unwrap();

                // V(s) <- π(a | s) * P(s' | s, a) * [R(s, a, s') + γV(s')]
                v += action_probability * transition_probability * (immediate_reward + (GAMMA * future_value));
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
    print_v_table(&v_table, (0, 0));

    // Train
    for i in 0..50 {
        policy_evaluation(&states, &actions, &mut v_table);

        println!("\nIteration {}", i + 1);
        print_v_table(&v_table, (0, 0));
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
