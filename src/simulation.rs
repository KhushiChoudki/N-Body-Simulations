use crate::{
    body::{Body, BodyType},
    brain::GNN,
    quadtree::{Quad, Quadtree},
    utils,
};

use broccoli::aabb::Rect;
use ultraviolet::Vec2;

pub struct Simulation {
    pub dt: f32,
    pub frame: usize,
    pub bodies: Vec<Body>,
    pub quadtree: Quadtree,
    pub warnings: Vec<(usize, usize, f32)>, // (index1, index2, distance)
    pub gnn: GNN,
}

impl Simulation {
    pub fn new() -> Self {
        let dt = 0.05;
        let n = 100000;
        let theta = 1.0;
        let epsilon = 1.0;

        let bodies: Vec<Body> = utils::uniform_disc(n);
        let quadtree = Quadtree::new(theta, epsilon);

        Self {
            dt,
            frame: 0,
            bodies,
            quadtree,
            warnings: Vec::new(),
            gnn: GNN::new(),
        }
    }

    pub fn step(&mut self) {
        self.iterate();
        self.collide();
        self.attract();
        self.solar_update();
        self.ml_predict();
        self.frame += 1;
    }

    pub fn attract(&mut self) {
        let quad = Quad::new_containing(&self.bodies);
        self.quadtree.clear(quad);

        for body in &self.bodies {
            if body.body_type != BodyType::Earth {
                self.quadtree.insert(body.pos, body.mass);
            }
        }

        self.quadtree.propagate();

        let earth = self.bodies.iter().find(|b| b.body_type == BodyType::Earth).cloned();

        for body in &mut self.bodies {
            // N-Body perturbations (Barnes-Hut)
            body.acc = self.quadtree.acc(body.pos);

            // Primary Earth gravity
            if let Some(earth) = &earth {
                if body.body_type != BodyType::Earth {
                    let d = earth.pos - body.pos;
                    let d_mag_sq = d.mag_sq();
                    if d_mag_sq > 1.0 {
                        let force = d * (earth.mass / (d_mag_sq * d_mag_sq.sqrt()));
                        body.acc += force;
                    }
                }
            }
        }
    }

    pub fn iterate(&mut self) {
        for body in &mut self.bodies {
            body.update(self.dt);
        }
    }

    pub fn collide(&mut self) {
        let mut rects = self
            .bodies
            .iter()
            .enumerate()
            .map(|(index, body)| {
                let pos = body.pos;
                let radius = body.radius;
                let min = pos - Vec2::one() * radius * 10.0; // Influence area for warnings
                let max = pos + Vec2::one() * radius * 10.0;
                (Rect::new(min.x, max.x, min.y, max.y), index)
            })
            .collect::<Vec<_>>();

        self.warnings.clear();
        let mut broccoli = broccoli::Tree::new(&mut rects);

        broccoli.find_colliding_pairs(|i, j| {
            let i = *i.unpack_inner();
            let j = *j.unpack_inner();

            let b1 = &self.bodies[i];
            let b2 = &self.bodies[j];
            let d = (b1.pos - b2.pos).mag();
            let r = b1.radius + b2.radius;

            if d < r {
                self.resolve(i, j);
            } else if d < r * 5.0 {
                // Warning: Close approach
                if b1.body_type == BodyType::Satellite || b2.body_type == BodyType::Satellite {
                    self.warnings.push((i, j, d));
                }
            }
        });
    }

    fn resolve(&mut self, i: usize, j: usize) {
        let b1 = &self.bodies[i];
        let b2 = &self.bodies[j];

        let p1 = b1.pos;
        let p2 = b2.pos;

        let r1 = b1.radius;
        let r2 = b2.radius;

        let d = p2 - p1;
        let r = r1 + r2;

        if d.mag_sq() > r * r {
            return;
        }

        let v1 = b1.vel;
        let v2 = b2.vel;

        let v = v2 - v1;

        let d_dot_v = d.dot(v);

        let m1 = b1.mass;
        let m2 = b2.mass;

        let weight1 = m2 / (m1 + m2);
        let weight2 = m1 / (m1 + m2);

        if d_dot_v >= 0.0 && d != Vec2::zero() {
            let tmp = d * (r / d.mag() - 1.0);
            self.bodies[i].pos -= weight1 * tmp;
            self.bodies[j].pos += weight2 * tmp;
            return;
        }

        let v_sq = v.mag_sq();
        let d_sq = d.mag_sq();
        let r_sq = r * r;

        let t = (d_dot_v + (d_dot_v * d_dot_v - v_sq * (d_sq - r_sq)).max(0.0).sqrt()) / v_sq;

        self.bodies[i].pos -= v1 * t;
        self.bodies[j].pos -= v2 * t;

        let p1 = self.bodies[i].pos;
        let p2 = self.bodies[j].pos;
        let d = p2 - p1;
        let d_dot_v = d.dot(v);
        let d_sq = d.mag_sq();

        let tmp = d * (1.5 * d_dot_v / d_sq);
        let v1 = v1 + tmp * weight1;
        let v2 = v2 - tmp * weight2;

        self.bodies[i].vel = v1;
        self.bodies[j].vel = v2;
        self.bodies[i].pos += v1 * t;
        self.bodies[j].pos += v2 * t;
    }

    pub fn solar_update(&mut self) {
        // Assume Sun is far away in the +X direction
        let sun_dir = Vec2::new(1.0, 0.0);
        let earth = self.bodies.iter().find(|b| b.body_type == BodyType::Earth).cloned();

        for body in &mut self.bodies {
            if body.body_type != BodyType::Satellite {
                body.exposure = 0.0;
                continue;
            }

            let mut exposure = 1.0;

            // Check for Earth shadow (simplified circle shadow)
            if let Some(earth) = &earth {
                let to_body = body.pos - earth.pos;
                let projection = to_body.dot(sun_dir);

                // If body is "behind" earth relative to sun
                if projection < 0.0 {
                    let closest_point_on_sun_axis = sun_dir * projection;
                    let dist_to_axis_sq = (to_body - closest_point_on_sun_axis).mag_sq();

                    if dist_to_axis_sq < earth.radius * earth.radius {
                        exposure = 0.0; // In shadow
                    }
                }
            }

            body.exposure = exposure;
            body.energy += exposure * self.dt * 0.1; // Arbitrary energy rate
        }
    }

    pub fn ml_predict(&mut self) {
        // In a real GNN, we'd use the spatial indices to find neighbors.
        // For this demo, we'll predict the state of the first 10 satellites.
        for i in 0..self.bodies.len().min(11) {
            let body = self.bodies[i];
            if body.body_type == BodyType::Satellite {
                // Placeholder for neighbor collection logic
                let neighbors = vec![]; 
                let (d_pos, d_vel) = self.gnn.predict(
                    body.pos, 
                    body.vel, 
                    body.mass, 
                    0.0, // BodyType::Satellite as float
                    &neighbors
                );
                
                // For demonstration, the GNN prediction is stored in a way the UI could potentially show "ghost" paths.
                // In this simplified version, we just run the forward pass to verify functionality.
            }
        }
    }
}
