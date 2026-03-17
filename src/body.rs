use ultraviolet::Vec2;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    Satellite,
    Debris,
    Earth,
}

#[derive(Clone, Copy)]
pub struct Body {
    pub pos: Vec2,
    pub vel: Vec2,
    pub acc: Vec2,
    pub mass: f32,
    pub radius: f32,
    pub body_type: BodyType,
    pub energy: f32,
    pub exposure: f32,
}

impl Body {
    pub fn new(pos: Vec2, vel: Vec2, mass: f32, radius: f32, body_type: BodyType) -> Self {
        Self {
            pos,
            vel,
            acc: Vec2::zero(),
            mass,
            radius,
            body_type,
            energy: 0.0,
            exposure: 0.0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.vel += self.acc * dt;
        self.pos += self.vel * dt;
    }
}
