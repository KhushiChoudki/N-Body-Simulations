use crate::body::{Body, BodyType};
use ultraviolet::Vec2;

pub fn uniform_disc(n: usize) -> Vec<Body> {
    fastrand::seed(0);
    let inner_radius = 50.0;
    let outer_radius = (n as f32).sqrt() * 6.0;

    let mut bodies: Vec<Body> = Vec::with_capacity(n);

    // Earth at the center
    let earth_mass = 2e7;
    let earth = Body::new(Vec2::zero(), Vec2::zero(), earth_mass as f32, inner_radius, BodyType::Earth);
    bodies.push(earth);

    let satellite_count = (n as f32 * 0.1) as usize; // 10% are satellites

    while bodies.len() < n {
        let a = fastrand::f32() * std::f32::consts::TAU;
        let (sin, cos) = a.sin_cos();
        let t = inner_radius / outer_radius;
        let r = fastrand::f32() * (1.0 - t * t) + t * t;
        let pos = Vec2::new(cos, sin) * outer_radius * r.sqrt();
        let vel = Vec2::new(sin, -cos);
        let mass = 1.0f32;
        let radius = mass.cbrt();

        let body_type = if bodies.len() <= satellite_count {
            BodyType::Satellite
        } else {
            BodyType::Debris
        };

        bodies.push(Body::new(pos, vel, mass, radius, body_type));
    }

    bodies.sort_by(|a, b| a.pos.mag_sq().total_cmp(&b.pos.mag_sq()));
    let mut mass = 0.0;
    for i in 0..n {
        mass += bodies[i].mass;
        if bodies[i].pos == Vec2::zero() {
            continue;
        }

        let v = (mass / bodies[i].pos.mag()).sqrt();
        bodies[i].vel *= v;
    }

    bodies
}
