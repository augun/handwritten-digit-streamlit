CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    predicted_digit INT,
    true_label INT,
    confidence FLOAT
);
