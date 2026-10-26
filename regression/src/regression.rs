const ALPHA: f32 = 0.01;

// Forward propagation
/*
 * Y' = XW + B
 */
fn forward(x: &Vec<f32>, w: f32, b: f32) -> Vec<f32> {
    x.iter()
        .map(|x_i| w * x_i + b)
        .collect()
}

// Cost function (Mean Squared Error)
/*
 * J = 1/n Σ_n (Y'_i - Y_i)^2
 */
fn cost(y_hat: &Vec<f32>, y: &Vec<f32>) -> f32 {
    let n = y.len() as f32;

    y_hat.iter()
        .zip(y.iter())
        .map(|(pred, actual)| (pred - actual).powi(2))
        .sum::<f32>() / n
}

// Backward propagation
/*
 * ∂J/∂W = 2/n Σ(Y' - Y)X
 * ∂J/∂B = 2/n Σ(Y' - Y)
 */
fn backward(x: &Vec<f32>, y_hat: &Vec<f32>, y: &Vec<f32>) -> (f32, f32) {
    let n = y.len() as f32;

    let mut dw = 0.0;
    let mut db = 0.0;

    for i in 0..y.len() {
        let error = y_hat[i] - y[i];

        dw += error * x[i];
        db += error;
    }

    dw = (2.0 / n) * dw;
    db = (2.0 / n) * db;

    (dw, db)
}

fn regression() {
    // Training dataset
    let x: Vec<f32> = vec![0.0, 1.0, 2.0, 4.0, 6.0, 8.0];
    let y: Vec<f32> = vec![4.0, 5.0, 6.0, 8.0, 10.0, 12.0];

    // Model parameters
    let mut w: f32 = 2.0;
    let mut b: f32 = 1.0;

    for epoch in 0..1000 {
        // Y' = XW + B
        let y_hat = forward(&x, w, b);

        // J -> 1/N Σ(Y' - Y)^2
        let loss = cost(&y_hat, &y);

        // ∂J/∂W and ∂J/∂B
        let (dw, db) = backward(&x, &y_hat, &y);

        // W <- W - α(∂J/∂W)
        // B <- B - α(∂J/∂B)
        w -= ALPHA * dw;
        b -= ALPHA * db;

        if epoch % 100 == 0 {
            println!("Epoch {epoch}: Loss = {loss:.6}");
        }
    }

    println!("Learned W = {w:.4}");
    println!("Learned B = {b:.4}");

    // Test
    let test_x = vec![10.0];
    let prediction = forward(&test_x, w, b);

    println!("Prediction for X = 10: {:?}", prediction);
}
