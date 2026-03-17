use ultraviolet::Vec2;
use fastrand;

#[derive(Clone)]
pub struct Linear {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut weights = Vec::with_capacity(in_features * out_features);
        let mut bias = Vec::with_capacity(out_features);

        let scale = (2.0 / in_features as f32).sqrt();
        for _ in 0..in_features * out_features {
            weights.push((fastrand::f32() - 0.5) * scale);
        }
        for _ in 0..out_features {
            bias.push(0.0);
        }

        Self {
            weights,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.out_features];
        for i in 0..self.out_features {
            let mut sum = self.bias[i];
            for j in 0..self.in_features {
                sum += input[j] * self.weights[i * self.in_features + j];
            }
            output[i] = sum.max(0.0); // ReLU activation
        }
        output
    }
}

pub struct GNN {
    pub node_enc: Linear,
    pub message_net: Linear,
    pub update_net: Linear,
}

impl GNN {
    pub fn new() -> Self {
        // Input: [pos.x, pos.y, vel.x, vel.y, mass, type]
        Self {
            node_enc: Linear::new(6, 16),
            message_net: Linear::new(16 * 2 + 2, 16), // [src, dst, rel_pos]
            update_net: Linear::new(16 + 16, 4),      // [enc, agg_msg] -> [d_pos, d_vel]
        }
    }

    pub fn predict(&self, pos: Vec2, vel: Vec2, mass: f32, body_type: f32, neighbors: &[(Vec2, Vec2, f32, f32)]) -> (Vec2, Vec2) {
        let node_features = [pos.x, pos.y, vel.x, vel.y, mass, body_type];
        let enc = self.node_enc.forward(&node_features);

        let mut agg_msg = vec![0.0; 16];
        if !neighbors.is_empty() {
            for (n_pos, n_vel, n_mass, n_type) in neighbors {
                let n_features = [n_pos.x, n_pos.y, n_vel.x, n_vel.y, *n_mass, *n_type];
                let n_enc = self.node_enc.forward(&n_features);
                
                let rel_pos = *n_pos - pos;
                let mut msg_in = Vec::with_capacity(34);
                msg_in.extend_from_slice(&enc);
                msg_in.extend_from_slice(&n_enc);
                msg_in.push(rel_pos.x);
                msg_in.push(rel_pos.y);
                
                let msg = self.message_net.forward(&msg_in);
                for i in 0..16 {
                    agg_msg[i] += msg[i];
                }
            }
            for i in 0..16 {
                agg_msg[i] /= neighbors.len() as f32;
            }
        }

        let mut update_in = Vec::with_capacity(32);
        update_in.extend_from_slice(&enc);
        update_in.extend_from_slice(&agg_msg);
        
        let update = self.update_net.forward(&update_in);
        
        (
            Vec2::new(update[0], update[1]),
            Vec2::new(update[2], update[3])
        )
    }
}
