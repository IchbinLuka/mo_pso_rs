# mo_pso_rs
An implementation of a multi-objective particle swarm optimization algorithm utilizing methods proposed in [Strategies for finding good local guides in multi-objective particle swarm optimization (MOPSO)](https://ieeexplore.ieee.org/document/1202243). 

This crate is mostly a Rust rewrite of the implementation of MOPSO in the [PSOTS toolbox](https://ieeexplore.ieee.org/document/5416861) with
additional support for constrained optimization with arbitrary constraints.

Currently, only two objectives are supported, but the implementation should be easily extendable to more objectives.


## Constraints
This implementation supports arbitrary constraints which are handled using a "Preserving feasibility technique" as described in the paper [Constraint-Handling Techniques for Particle Swarm Optimization Algorithms](https://arxiv.org/abs/2101.10933). 

## Example
```rust
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
```