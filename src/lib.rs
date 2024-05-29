use std::{cmp::Ordering, ops::Add};


#[derive(Clone)]
pub struct Particle {
    pub position: Vec<f64>,
    velocity: Vec<f64>,
    pbest: Vec<f64>,
    pbest_value: [f64; 2],
    pub value: [f64; 2],
    feasible: bool,
}

/// Returns true if `v1` is Pareto dominant over `v2`.
#[inline(always)]
fn is_pareto_dominant(v1: &[f64; 2], v2: &[f64; 2]) -> bool {
    (v1[0] <= v2[0] && v1[1] < v2[1]) || (v1[0] < v2[0] && v1[1] <= v2[1])
}

impl Particle {
    pub fn random(dimensions: usize) -> Particle {
        let position: Vec<f64> = std::iter::repeat(())
            .take(dimensions)
            .map(|_| 20.0 * rand::random::<f64>())
            .collect();
        Self {
            pbest: position.clone(),
            position,
            velocity: std::iter::repeat(())
                .take(dimensions)
                .map(|_| rand::random::<f64>())
                .collect(),
            pbest_value: [f64::INFINITY; 2],
            value: [f64::INFINITY; 2],
            feasible: false,
        }
    }

    pub fn new(position: Vec<f64>, velocity: Vec<f64>) -> Particle {
        Self {
            pbest: position.clone(),
            position,
            velocity,
            pbest_value: [f64::INFINITY; 2],
            value: [f64::INFINITY; 2],
            feasible: false,
        }
    }

    pub fn with_position(position: Vec<f64>) -> Particle {
        Self {
            pbest: position.clone(),
            position: position.clone(),
            velocity: std::iter::repeat(())
                .take(position.len())
                .map(|_| rand::random::<f64>())
                .collect(),
            pbest_value: [f64::INFINITY; 2],
            value: [f64::INFINITY; 2],
            feasible: false,
        }
    }

    fn update_pbest(&mut self) {
        if is_pareto_dominant(&self.value, &self.pbest_value) {
            self.pbest_value = self.value;
            self.pbest.clone_from(&self.position);
        }
    }
}

/// A trait that represents a multi-objective optimization problem.
pub trait Problem {
    /// The objective function of the problem.
    fn objective(&self, p: &[f64]) -> [f64; 2];

    /// Returns true if the given solution is feasible.
    fn is_feasible(&self, _p: &[f64]) -> bool {
        true
    }
}

pub struct MoPsoConfig {
    /// The amount of iterations to run the algorithm.
    pub max_iter: usize,
    /// The cognitive parameter.
    pub c_1: f64,
    /// The social parameter.
    pub c_2: f64,
    /// The initial inertia weight.
    pub w_start: f64,
    /// The final inertia weight.
    pub w_end: f64,
    /// The size of the archive used to store non-dominated solutions.
    pub archive_size: usize,
}

/// Multi-Objective Particle Swarm Optimization.
pub struct MoPso<'a, T: Problem> {
    particles: Vec<Particle>,
    archive: Vec<Particle>,
    config: MoPsoConfig,
    pub problem: &'a T,
}

impl<'a, T: Problem> MoPso<'a, T> {
    pub fn new(config: MoPsoConfig, problem: &'a T, particles: Vec<Particle>) -> Self {
        MoPso {
            particles,
            archive: Vec::with_capacity(config.archive_size),
            config,
            problem,
        }
    }

    #[inline]
    fn update_particle_values(&mut self) {
        for particle in &mut self.particles {
            particle.value = self.problem.objective(&particle.position);
            particle.feasible = self.problem.is_feasible(&particle.position);
        }
    }

    #[inline]
    fn update_particle_pbest(&mut self) {
        for particle in &mut self.particles {
            if particle.feasible {
                particle.update_pbest();
            }
        }
    }

    #[inline]
    fn update_archive(&mut self) {
        if self.archive.is_empty() {
            self.archive.push(
                self.particles
                    .first()
                    .expect("Swarm cannot be empty")
                    .clone(),
            );
            return;
        }

        for particle in &self.particles {
            if !particle.feasible {
                continue;
            }

            let mut particle_dominant = false;
            for archive_particle in &mut self.archive {
                if is_pareto_dominant(&particle.value, &archive_particle.value) {
                    *archive_particle = particle.clone();
                    particle_dominant = true;
                    break;
                }
            }

            if particle_dominant {
                continue;
            }

            // If archive has too many elements, delete the most similar ones.
            if self.archive.len() + 1 > self.config.archive_size {
                for _ in 0..(self.archive.len() - self.config.archive_size) {
                    let sigma = calc_sigma(&self.archive);
                    let mut best_item = (0, (sigma[0] - sigma[1]).abs());
                    for (i, s_1) in sigma.iter().enumerate() {
                        // TODO: We do not need to compare the same sigma values twice.
                        for (j, s_2) in sigma.iter().enumerate() {
                            if i == j {
                                continue;
                            }

                            let diff = (s_1 - s_2).abs();
                            if diff < best_item.1 {
                                best_item = (i, diff);
                            }
                        }
                    }
                    self.archive[best_item.0] = self.archive.last().unwrap().clone();
                    self.archive.pop();
                }
            }
            self.archive.push(particle.clone());
        }

        // Ensure that there are no solutions that are Pareto dominant over each other.
        // TODO: There must be a more efficient way to do this.
        loop {
            let mut modified = false;
            let mut archive_len = self.archive.len();
            let mut i = 0;
            while i < archive_len {
                let mut j = i + 1;
                while j < archive_len {
                    let value_i = self.archive[i].value;
                    let value_j = self.archive[j].value;
                    if value_i[0] == value_j[0]
                        || value_i[1] == value_j[1]
                        || is_pareto_dominant(&value_i, &value_j)
                    {
                        // Delete solution
                        let temp = self.archive.last().unwrap().clone();
                        self.archive[j] = temp;
                        self.archive.pop();
                        archive_len -= 1;
                        modified = true;
                    }

                    if j >= archive_len || i == archive_len {
                        break;
                    }
                    j += 1;
                }
                i += 1;
            }
            if !modified {
                break;
            }
        }
    }

    #[inline]
    fn calc_gbest(&self) -> Vec<Particle> {
        let sigma_archive = calc_sigma(&self.archive);
        let sigma_particles = calc_sigma(&self.particles);

        let mut result = Vec::with_capacity(self.particles.len());
        for (_, sigma) in self.particles.iter().zip(sigma_particles.iter()) {
            let (g_best, _) = self
                .archive
                .iter()
                .zip(sigma_archive.iter())
                .min_by_key(|(_, s)| Comparablef64((sigma - **s).abs()))
                .expect("Archive should not be empty");
            result.push(g_best.clone());
        }
        result
    }

    pub fn optimize(mut self) -> Vec<Particle> {
        let mut iter = 0;
        while iter < self.config.max_iter {
            self.update_particle_values();
            self.update_particle_pbest();
            self.update_archive();

            // Calculate the inertia weight
            let w = (self.config.w_start - self.config.w_end)
                * ((self.config.max_iter - iter) as f64 / self.config.max_iter as f64)
                + self.config.w_end;

            let g_best = self.calc_gbest();

            for (particle, g_best) in self.particles.iter_mut().zip(g_best.iter()) {
                // Update velocity
                for (i, ((v, p), g)) in particle
                    .velocity
                    .iter_mut()
                    .zip(particle.position.iter())
                    .zip(g_best.position.iter())
                    .enumerate()
                {
                    let r1 = rand::random::<f64>();
                    let r2 = rand::random::<f64>();
                    *v = w * *v
                        + self.config.c_1 * r1 * (particle.pbest[i] - p)
                        + self.config.c_2 * r2 * (g - p);
                }

                // Update position
                for (p, v) in particle.position.iter_mut().zip(particle.velocity.iter()) {
                    *p += *v;
                }
            }

            iter += 1;
        }
        self.archive
            .iter()
            .filter(|p| p.feasible)
            .cloned()
            .collect()
    }
}

fn calc_sigma(particles: &[Particle]) -> Vec<f64> {
    particles
        .iter()
        .map(|p| {
            let p_sum = p.position[0].powi(2) + p.position[1].powi(2);
            if p_sum == 0.0 {
                0.0
            } else {
                (p.position[0].powi(2) - p.position[1].powi(2)) / p_sum
            }
        })
        .collect()
}

/// A wrapper type for `f32` that implements `PartialOrd` and `Ord` traits.
///
/// Note: since we are dealing with floating point numbers, this may not always work as expected.
#[derive(PartialEq, Clone, Copy)]
pub struct Comparablef64(pub f64);

impl Add for Comparablef64 {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Comparablef64(self.0 + other.0)
    }
}

impl Eq for Comparablef64 {}

impl PartialOrd for Comparablef64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Comparablef64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

impl From<f64> for Comparablef64 {
    fn from(value: f64) -> Self {
        Self(value)
    }
}
