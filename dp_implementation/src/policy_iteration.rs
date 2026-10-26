use std::collections::HashMap;

use crate::snake::{HEIGHT, WIDTH};
use crate::snake::mdp::{State, Action, world_model, reward, GAMMA};


// Policy (π) - learnt
fn pi_table(states: &Vec<State>, actions: &Vec<Action>) -> HashMap<(State, Action), f32> {
    let mut table: HashMap<(State, Action), f32> = HashMap::new();

    for s in states {
        for a in actions {
            table.insert((s.clone(), a.clone()), 1.0 / actions.len() as f32);
        }
    }

    table
}

pub fn policy(s: &State, a: &Action, pi_table: &mut HashMap<(State, Action), f32>) -> f32 {
    let sa = (s.clone(), a.clone());
    pi_table[&sa]
}

// Policy iteration
/* Has two parts
 *   - Policy evauluation
 *   - Policy improvement
 *
 * Policy is improved by calculating the Q(s,a) for every possible action a and
 *   updating the policy table with the action a* (action a with best Q value)
 *
 *     Q(s, a) = Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
 *     a* = argmax_a Q(s, a)
 *     π(a | s) <- 1 if a == a*; 0 otherwise                    // greedy/deterministic
 *     π(a | s) <- 0.9 if a == a*; 0.1 * (1 / n - 1) otherwise  // stochastic
 */
fn policy_iteration(
    states: &Vec<State>,
    actions: &Vec<Action>,
    pi_table: &mut HashMap<(State, Action), f32>,
    v_table: &mut HashMap<State, f32>
) {
    // Policy evaluation
    for _ in 0..5 {
        policy_evaluation(&states, &actions, pi_table, v_table);
    }

    // Policy improvement

    let mut new_table = HashMap::new();

    for s in states {
        let mut best_action = actions[0].clone(); // a*
        let mut best_action_value = f32::NEG_INFINITY;   // max_a Q(s, a)

        for a in actions {
            // Q(s, a)
            let mut action_value = 0.0;

            // Σ_s'
            for s_next in states {

                // P(s' | s,a)
                let transition_probability = world_model(s, a, s_next);

                // R(s, a, s')
                let immediate_reward = reward(s, a, s_next);

                // V(s')
                let future_value = value(s_next, v_table);

                // Q(s, a) = P(s' | s, a) * [R(s, a, s') + γV(s')]
                action_value += transition_probability * (immediate_reward + (GAMMA * future_value));
            }

            // a* <- arg_max_a Q(s, a)
            if action_value > best_action_value {
                best_action = a.clone();   // a*
                best_action_value = action_value; // max_a Q(s, a)
            }
        }

        // π(a | s) <- 1 if a == a*, 0 otherwise
        for a in actions {
            if *a == best_action {
                new_table.insert((s.clone(), a.clone()), 1.0);
            } else {
                new_table.insert((s.clone(), a.clone()), 0.0);
            };
        }
    }

    *pi_table = new_table;
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
 * Value is learnt by continuously interacting with the world_model
 *
 *     V(s) <- Σ_a π(a | s) Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
 */
pub fn policy_evaluation(
    states: &Vec<State>,
    actions: &Vec<Action>,
    pi_table: &mut HashMap<(State, Action), f32>,
    v_table: &mut HashMap<State, f32>
) {
    let mut new_table = HashMap::new();

    for s in states {
        // V(s)
        let mut v = 0.0;

        // Σ_a
        for a in actions {
            // π(a | s)
            let action_probability = policy(s, a, pi_table);

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
    let mut pi_table = pi_table(&states, &actions);
    let mut v_table = v_table(&states);

    println!("\nInitial table");
    print_v_table(&v_table, (0, 3));

    // Train
    for i in 0..50 {
        policy_iteration(&states, &actions, &mut pi_table, &mut v_table);

        println!("\nIteration {}", i + 1);
        print_v_table(&v_table, (0, 3));
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
