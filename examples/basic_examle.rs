use mo_pso_rs::{MoPso, MoPsoConfig, Particle, Problem};

struct TestProblem {
    radius: f64
}
impl Problem for TestProblem {
    fn objective(&self, p: &[f64]) -> [f64; 2] {
        [p[0].powi(2), -p[1].powi(2)]
    }

    fn is_feasible(&self, p: &[f64]) -> bool {
        // Only allow point outside circle with radius self.radius
        p[0].powi(2) + p[1].powi(2) >= self.radius.powi(2)
    }
}


fn main() {
    let problem = TestProblem { radius: 2.0 };
    
    let config = MoPsoConfig {
        max_iter: 100,
        c_1: 0.4, 
        c_2: 0.4,
        w_start: 0.9,
        w_end: 0.4,
        archive_size: 100,
    };

    let mut particles = Vec::new();
    for _ in 0..100 {
        particles.push(Particle::random(2));
    }

    let optimizer = MoPso::new(config, &problem, particles);
    let pareto_front = optimizer.optimize();

    for particle in pareto_front {
        println!("{:?}", particle.position);
    }
}