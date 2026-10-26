use super::{HEIGHT, WIDTH};


// State (S)
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct State {
    pub head: (i32, i32),
    pub fruit: (i32, i32),
}

impl State {
    pub fn all() -> Vec<State> {
        let mut states = Vec::<State>::with_capacity(HEIGHT as usize * WIDTH as usize);

        for h_y in 0..HEIGHT {
            for h_x in 0..WIDTH {
                for f_y in 0..HEIGHT {
                    for f_x in 0..WIDTH {
                        states.push(
                            State {
                                head: (h_x, h_y),
                                fruit: (f_x, f_y),
                            }
                        );
                    }
                }
            }
        }

        states
    }

    pub fn is_terminal(&self) -> bool {
        self.head == self.fruit
    }
}


// Action (A)
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
    Eat,
}

impl Action {
    pub fn all() -> Vec<Action> {
        Vec::from([
            Action::Up,
            Action::Down,
            Action::Left,
            Action::Right,
            Action::Eat,
        ])
    }
}


// World model (P)
pub fn world_model(s: &State, a: &Action, s_next: &State) -> f32 {
    if s.fruit != s_next.fruit {
        return 0.0;
    }

    let x_diff = s.head.0 - s_next.head.0;
    let y_diff = s.head.1 - s_next.head.1 ;

    match a {
        Action::Up => {
            if x_diff != 0 {
                0.0
            } else if y_diff == 1 || (y_diff == 0 && s.head.1 == 0) {
                1.0
            } else {
                0.0
            }
        },
        Action::Down => {
            if x_diff != 0 {
                0.0
            } else if y_diff == -1 || (y_diff == 0 && s.head.1 == HEIGHT - 1) {
                1.0
            } else {
                0.0
            }
        },
        Action::Left => {
            if y_diff != 0 {
                0.0
            } else if x_diff == 1 || (x_diff == 0 && s.head.0 == 0) {
                1.0
            } else {
                0.0
            }
        },
        Action::Right => {
            if y_diff != 0 {
                0.0
            } else if x_diff == -1 || (x_diff == 0 && s.head.0 == WIDTH - 1) {
                1.0
            } else {
                0.0
            }
        },
        Action::Eat => {
            if (x_diff, y_diff) == (0, 0) {
                1.0
            } else {
                0.0
            }
        }
    }
}


// Reward (R)
pub fn reward(s: &State, a: &Action, s_next: &State) -> f32 {
    match a {
        Action::Eat => {
            if (s_next.head == s_next.fruit) && (s.fruit == s_next.fruit) {
                1.0
            } else {
                -2.0
            }
        },
        _ => {
            // Hit a wall
            if s == s_next {
                -2.0
            // Normal movement
            } else {
                -1.0
            }
        }
    }
}


// Gamma (γ)
pub const GAMMA: f32 = 0.9;
