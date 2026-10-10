//! HIVE BRAIN: 32 candle neural networks (CPU) fight to the death.
//!
//! `cargo run` does this:
//!   1. quick silent training so the brains are not random (evolution)
//!   2. a tournament, watched live in a terminal UI (ratatui):
//!        ROUND 1: all 32 fight  -> top 25 advance
//!        ROUND 2: 25 fight      -> top 15 advance
//!        ROUND 3: 15 fight      -> top 10 advance
//!        FINAL  : 10 fight      -> one champion
//!
//! Flags (after `--`):  --train N  --rounds N  --seed N  --headless  --snapshot

use candle_core::{Device, Result as CResult, Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use ratatui::{
    Frame, Terminal,
    backend::TestBackend,
    crossterm::event::{self, Event as CEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{
        Block, List, ListItem, Paragraph,
        canvas::{Canvas, Circle},
    },
};
use std::collections::HashSet;
use std::io::{IsTerminal, Write};
use std::time::Duration;

type R<T> = Result<T, Box<dyn std::error::Error>>;

// ---------- sizes ----------
const POP: usize = 32; // 32 fighters
const IN: usize = 11; // sensors
const H1: usize = 24;
const H2: usize = 16;
const OUT: usize = 4; // move_x, move_y, attack, shield
const GENOME_LEN: usize = IN * H1 + H1 + H1 * H2 + H2 + H2 * OUT + OUT;

// ---------- arena rules ----------
const ARENA_R: f32 = 70.0;
const RANGE: f32 = 14.0;
const SPEED: f32 = 2.2;
const ATTACK_DMG: f32 = 14.0;
const ATTACK_COST: f32 = 10.0;
const COOLDOWN: u32 = 4;
const STORM_START: u32 = 300;
const MAX_TICKS: u32 = 1500;

const SYMS: &[u8; POP] = b"0123456789abcdefghijklmnopqrstuv";

/// (fighters that advance, label)
const STAGES: [(usize, &str); 4] = [
    (25, "ROUND 1"),
    (15, "ROUND 2"),
    (10, "ROUND 3"),
    (1, "FINAL"),
];
const SPEEDS: [u32; 6] = [1, 2, 4, 8, 16, 32];

// ---------- options ----------
struct Opts {
    train: usize,
    rounds: usize,
    seed: u64,
    headless: bool,
    snapshot: bool,
}

fn val<T: std::str::FromStr>(args: &[String], i: &mut usize) -> Option<T> {
    *i += 1;
    args.get(*i).and_then(|s| s.parse().ok())
}

fn parse_opts() -> Opts {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut o = Opts {
        train: 60,
        rounds: 2,
        seed: 42,
        headless: false,
        snapshot: false,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--train" => o.train = val::<usize>(&args, &mut i).unwrap_or(60),
            "--rounds" => o.rounds = val::<usize>(&args, &mut i).unwrap_or(2).max(1),
            "--seed" => o.seed = val::<u64>(&args, &mut i).unwrap_or(42),
            "--headless" => o.headless = true,
            "--snapshot" => o.snapshot = true,
            _ => {}
        }
        i += 1;
    }
    o
}

// ---------- the brain (candle, CPU) ----------
struct Brain {
    dev: Device,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,
}

impl Brain {
    fn from_genome(g: &[f32]) -> CResult<Self> {
        let dev = Device::Cpu;
        let mut o = 0usize;
        let mut take = |n: usize, shape: (usize, usize)| -> CResult<Tensor> {
            let t = Tensor::from_slice(&g[o..o + n], shape, &dev)?;
            o += n;
            Ok(t)
        };
        let w1 = take(IN * H1, (IN, H1))?;
        let b1 = take(H1, (1, H1))?;
        let w2 = take(H1 * H2, (H1, H2))?;
        let b2 = take(H2, (1, H2))?;
        let w3 = take(H2 * OUT, (H2, OUT))?;
        let b3 = take(OUT, (1, OUT))?;
        Ok(Self {
            dev: Device::Cpu,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        })
    }

    fn think(&self, obs: &[f32; IN]) -> CResult<[f32; OUT]> {
        let x = Tensor::from_slice(&obs[..], (1, IN), &self.dev)?;
        let h = x.matmul(&self.w1)?.broadcast_add(&self.b1)?.tanh()?;
        let h = h.matmul(&self.w2)?.broadcast_add(&self.b2)?.tanh()?;
        let y = h.matmul(&self.w3)?.broadcast_add(&self.b3)?.tanh()?;
        let v = y.flatten_all()?.to_vec1::<f32>()?;
        Ok([v[0], v[1], v[2], v[3]])
    }
}

// ---------- fighter ----------
struct Fighter {
    uid: usize,  // permanent name, survives across generations (elites keep it)
    born: usize, // generation this brain was born
    career_wins: u32,
    career_kills: u32,
    genome: Vec<f32>,
    brain: Brain,
    x: f32,
    y: f32,
    hp: f32,
    energy: f32,
    cooldown: u32,
    alive: bool,
    kills: u32,
    dmg_dealt: f32,
    survived: u32,
    storm_ticks: u32,
    fitness: f32, // this fight
    score: f32,   // summed over a training generation's rounds
}

impl Fighter {
    fn new(uid: usize, born: usize, wins: u32, kills: u32, genome: Vec<f32>) -> CResult<Self> {
        let brain = Brain::from_genome(&genome)?;
        Ok(Self {
            uid,
            born,
            career_wins: wins,
            career_kills: kills,
            genome,
            brain,
            x: 0.0,
            y: 0.0,
            hp: 100.0,
            energy: 100.0,
            cooldown: 0,
            alive: true,
            kills: 0,
            dmg_dealt: 0.0,
            survived: 0,
            storm_ticks: 0,
            fitness: 0.0,
            score: 0.0,
        })
    }

    fn name(&self) -> String {
        format!("F{}(g{})", self.uid, self.born)
    }

    fn reset(&mut self, x: f32, y: f32) {
        self.x = x;
        self.y = y;
        self.hp = 100.0;
        self.energy = 100.0;
        self.cooldown = 0;
        self.alive = true;
        self.kills = 0;
        self.dmg_dealt = 0.0;
        self.survived = 0;
        self.storm_ticks = 0;
        self.fitness = 0.0;
    }
}

fn gauss(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.gen_range(1e-7..1.0);
    let u2: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn random_genome(rng: &mut StdRng) -> Vec<f32> {
    (0..GENOME_LEN).map(|_| gauss(rng) * 0.5).collect()
}

fn dist(a: &Fighter, b: &Fighter) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

fn nearest_enemy(pop: &[Fighter], i: usize) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for j in 0..pop.len() {
        if j == i || !pop[j].alive {
            continue;
        }
        let d = dist(&pop[i], &pop[j]);
        if best.map_or(true, |(_, bd)| d < bd) {
            best = Some((j, d));
        }
    }
    best.map(|(j, _)| j)
}

fn observe(pop: &[Fighter], i: usize, storm_r: f32) -> [f32; IN] {
    let me = &pop[i];
    let (dx, dy, d, ehp) = match nearest_enemy(pop, i) {
        Some(j) => {
            let e = &pop[j];
            let d = dist(me, e);
            (
                (e.x - me.x) / ARENA_R,
                (e.y - me.y) / ARENA_R,
                d / ARENA_R,
                e.hp / 100.0,
            )
        }
        None => (0.0, 0.0, 1.0, 0.0),
    };
    let in_range = pop
        .iter()
        .enumerate()
        .filter(|(j, o)| *j != i && o.alive && dist(me, o) <= RANGE)
        .count() as f32;
    let me_r = (me.x * me.x + me.y * me.y).sqrt();
    [
        me.hp / 100.0,
        me.energy / 100.0,
        me.x / ARENA_R,
        me.y / ARENA_R,
        dx,
        dy,
        d,
        ehp,
        if me.cooldown == 0 { 1.0 } else { 0.0 },
        in_range / 4.0,
        (storm_r - me_r) / ARENA_R, // distance to the deadly edge
    ]
    .map(|v| v.clamp(-1.5, 1.5))
}

// ---------- kill feed ----------
#[derive(Clone)]
struct KillEvent {
    tick: u32,
    killer: Option<String>,
    victim: String,
    x: f32,
    y: f32,
}

fn zone(x: f32, y: f32) -> String {
    let r = (x * x + y * y).sqrt();
    if r < 15.0 {
        return "the center".to_string();
    }
    const NAMES: [&str; 8] = [
        "East",
        "North-East",
        "North",
        "North-West",
        "West",
        "South-West",
        "South",
        "South-East",
    ];
    let idx = ((y.atan2(x) / (std::f32::consts::PI / 4.0)).round() as i32).rem_euclid(8) as usize;
    format!("{}, {:.0}m from center", NAMES[idx], r)
}

fn fmt_event(e: &KillEvent) -> String {
    match &e.killer {
        Some(k) => format!(
            "t={:>4}  {} KILLED {}  at ({:+.0},{:+.0}) {}",
            e.tick,
            k,
            e.victim,
            e.x,
            e.y,
            zone(e.x, e.y)
        ),
        None => format!(
            "t={:>4}  THE STORM killed {}  at ({:+.0},{:+.0}) {}",
            e.tick,
            e.victim,
            e.x,
            e.y,
            zone(e.x, e.y)
        ),
    }
}

// ---------- one fight, advanced tick by tick ----------
struct Battle {
    pop: Vec<Fighter>,
    t: u32,
    storm_r: f32,
    events: Vec<KillEvent>,
    attacked: Vec<bool>,
    was_hit: Vec<bool>,
    shielding: Vec<bool>,
    done: bool,
    winner: Option<usize>,
}

impl Battle {
    fn new(mut pop: Vec<Fighter>, rng: &mut StdRng) -> Self {
        let n = pop.len();
        let mut spots: Vec<(f32, f32)> = (0..n)
            .map(|i| {
                let a = (i as f32 / n as f32) * std::f32::consts::TAU + rng.gen_range(-0.05..0.05);
                let r = 50.0 + rng.gen_range(-5.0..5.0);
                (a.cos() * r, a.sin() * r)
            })
            .collect();
        spots.shuffle(rng); // nobody gets a lucky starting position every time
        for (f, (x, y)) in pop.iter_mut().zip(spots) {
            f.reset(x, y);
        }
        Self {
            pop,
            t: 0,
            storm_r: ARENA_R,
            events: Vec::new(),
            attacked: vec![false; n],
            was_hit: vec![false; n],
            shielding: vec![false; n],
            done: false,
            winner: None,
        }
    }

    fn alive(&self) -> usize {
        self.pop.iter().filter(|f| f.alive).count()
    }

    fn step(&mut self, rng: &mut StdRng) -> CResult<()> {
        if self.done {
            return Ok(());
        }
        let n = self.pop.len();
        if self.alive() <= 1 || self.t >= MAX_TICKS {
            self.finish();
            return Ok(());
        }
        self.t += 1;
        let t = self.t;
        let storm_r = if t > STORM_START {
            (ARENA_R - (t - STORM_START) as f32 * 0.07).max(8.0)
        } else {
            ARENA_R
        };
        self.storm_r = storm_r;

        // 1. everyone thinks (candle forward pass)
        let mut acts = vec![[0.0f32; OUT]; n];
        for i in 0..n {
            if !self.pop[i].alive {
                continue;
            }
            let obs = observe(&self.pop, i, storm_r);
            acts[i] = self.pop[i].brain.think(&obs)?;
        }

        // 2. movement + shields
        self.shielding = vec![false; n];
        for i in 0..n {
            if !self.pop[i].alive {
                continue;
            }
            let f = &mut self.pop[i];
            f.x += acts[i][0] * SPEED;
            f.y += acts[i][1] * SPEED;
            let r = (f.x * f.x + f.y * f.y).sqrt();
            if r > ARENA_R {
                f.x *= ARENA_R / r;
                f.y *= ARENA_R / r;
            }
            f.energy -= 0.15 * (acts[i][0].abs() + acts[i][1].abs());
            if acts[i][3] > 0.3 && f.energy >= 1.0 {
                self.shielding[i] = true;
                f.energy -= 0.8;
            }
        }

        // 3. attacks, random order
        self.attacked = vec![false; n];
        self.was_hit = vec![false; n];
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(rng);
        for &i in &order {
            if !self.pop[i].alive {
                continue;
            }
            if acts[i][2] > 0.3 && self.pop[i].cooldown == 0 && self.pop[i].energy >= ATTACK_COST {
                if let Some(j) = nearest_enemy(&self.pop, i) {
                    if dist(&self.pop[i], &self.pop[j]) <= RANGE {
                        let dmg = if self.shielding[j] {
                            ATTACK_DMG * 0.25
                        } else {
                            ATTACK_DMG
                        };
                        self.pop[j].hp -= dmg;
                        self.pop[i].dmg_dealt += dmg;
                        self.pop[i].energy -= ATTACK_COST;
                        self.pop[i].cooldown = COOLDOWN;
                        self.attacked[i] = true;
                        self.was_hit[j] = true;
                        if self.pop[j].hp <= 0.0 && self.pop[j].alive {
                            self.pop[j].alive = false;
                            self.pop[j].survived = t;
                            self.pop[i].kills += 1;
                            self.events.push(KillEvent {
                                tick: t,
                                killer: Some(self.pop[i].name()),
                                victim: self.pop[j].name(),
                                x: self.pop[j].x,
                                y: self.pop[j].y,
                            });
                        }
                    }
                }
            }
        }

        // 4. storm, regen, cooldowns
        for i in 0..n {
            if !self.pop[i].alive {
                continue;
            }
            let f = &mut self.pop[i];
            let r = (f.x * f.x + f.y * f.y).sqrt();
            if r > storm_r {
                f.hp -= 2.5;
                f.storm_ticks += 1;
            }
            f.energy = (f.energy + 1.2).min(100.0);
            if f.cooldown > 0 {
                f.cooldown -= 1;
            }
            if f.hp <= 0.0 {
                f.alive = false;
                f.survived = t;
                let (x, y) = (f.x, f.y);
                let victim = self.pop[i].name();
                self.events.push(KillEvent {
                    tick: t,
                    killer: None,
                    victim,
                    x,
                    y,
                });
            }
        }
        Ok(())
    }

    fn finish(&mut self) {
        self.done = true;
        let t = self.t;
        for f in self.pop.iter_mut() {
            if f.alive {
                f.survived = t;
            }
        }
        let alive: Vec<usize> = (0..self.pop.len()).filter(|&i| self.pop[i].alive).collect();
        self.winner = if alive.len() == 1 {
            Some(alive[0])
        } else {
            None
        };
        let winner = self.winner;
        for (i, f) in self.pop.iter_mut().enumerate() {
            f.fitness = f.kills as f32 * 50.0 + f.dmg_dealt * 0.4 + f.survived as f32 * 0.02
                - f.storm_ticks as f32 * 0.5
                + if winner == Some(i) { 150.0 } else { 0.0 };
        }
    }

    fn run_to_end(&mut self, rng: &mut StdRng) -> CResult<()> {
        while !self.done {
            self.step(rng)?;
        }
        Ok(())
    }
}

// ---------- evolution (quick silent training) ----------
struct Seed {
    uid: usize,
    born: usize,
    wins: u32,
    kills: u32,
    genome: Vec<f32>,
}

fn evolve(
    pop: &mut Vec<Fighter>,
    rng: &mut StdRng,
    generation: usize,
    next_uid: &mut usize,
) -> CResult<()> {
    let mut idx: Vec<usize> = (0..POP).collect();
    idx.sort_by(|&a, &b| pop[b].score.partial_cmp(&pop[a].score).unwrap());

    let elites = 6;
    let parents = 12;
    let immigrants = 2;
    let sigma = (0.25 * 0.995f32.powi(generation as i32)).max(0.04);

    let mut next: Vec<Seed> = Vec::with_capacity(POP);
    for &i in idx.iter().take(elites) {
        let f = &pop[i];
        next.push(Seed {
            uid: f.uid,
            born: f.born,
            wins: f.career_wins,
            kills: f.career_kills,
            genome: f.genome.clone(),
        });
    }
    while next.len() < POP - immigrants {
        let a = &pop[idx[rng.gen_range(0..parents)]].genome;
        let b = &pop[idx[rng.gen_range(0..parents)]].genome;
        let mut child: Vec<f32> = (0..GENOME_LEN)
            .map(|k| if rng.gen_bool(0.5) { a[k] } else { b[k] })
            .collect();
        for w in child.iter_mut() {
            if rng.gen_bool(0.12) {
                *w += gauss(rng) * sigma;
            }
        }
        next.push(Seed {
            uid: *next_uid,
            born: generation + 1,
            wins: 0,
            kills: 0,
            genome: child,
        });
        *next_uid += 1;
    }
    while next.len() < POP {
        next.push(Seed {
            uid: *next_uid,
            born: generation + 1,
            wins: 0,
            kills: 0,
            genome: random_genome(rng),
        });
        *next_uid += 1;
    }

    *pop = next
        .into_iter()
        .map(|s| Fighter::new(s.uid, s.born, s.wins, s.kills, s.genome))
        .collect::<CResult<Vec<_>>>()?;
    Ok(())
}

fn train(opts: &Opts, rng: &mut StdRng, show_progress: bool) -> R<Vec<Fighter>> {
    let mut pop: Vec<Fighter> = (0..POP)
        .map(|i| Fighter::new(i, 0, 0, 0, random_genome(rng)))
        .collect::<CResult<_>>()?;
    let mut next_uid = POP;
    for g in 0..opts.train {
        for f in pop.iter_mut() {
            f.score = 0.0;
        }
        for _ in 0..opts.rounds {
            let mut b = Battle::new(pop, rng);
            b.run_to_end(rng)?;
            let w = b.winner;
            pop = b.pop;
            for f in pop.iter_mut() {
                f.score += f.fitness;
                f.career_kills += f.kills;
            }
            if let Some(w) = w {
                pop[w].career_wins += 1;
            }
        }
        if show_progress {
            let done = (g + 1) * 30 / opts.train;
            eprint!(
                "\r  training the brains  [{}{}] {}/{}",
                "#".repeat(done),
                "-".repeat(30 - done),
                g + 1,
                opts.train
            );
            std::io::stderr().flush().ok();
        }
        if g + 1 < opts.train {
            evolve(&mut pop, rng, g, &mut next_uid)?;
        }
    }
    if show_progress && opts.train > 0 {
        eprintln!();
    }
    // fresh scores/stats for the tournament
    for f in pop.iter_mut() {
        f.score = 0.0;
    }
    Ok(pop)
}

// ---------- tournament bookkeeping ----------
#[derive(Clone)]
struct Row {
    name: String,
    kills: u32,
    dmg: f32,
    fitness: f32,
    advances: bool,
    wins: u32,
    career_kills: u32,
}

struct StageRecord {
    label: String,
    from: usize,
    keep: usize,
    ticks: u32,
    rows: Vec<Row>, // best first
    events: Vec<KillEvent>,
}

/// Rank a finished fight, cut to the top `keep`, return the survivors.
fn conclude(mut b: Battle, keep: usize, label: &str) -> (Vec<Fighter>, StageRecord) {
    for f in b.pop.iter_mut() {
        f.career_kills += f.kills;
    }
    if let Some(w) = b.winner {
        b.pop[w].career_wins += 1;
    }
    let n = b.pop.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &c| b.pop[c].fitness.partial_cmp(&b.pop[a].fitness).unwrap());
    let advance: HashSet<usize> = idx.iter().take(keep).cloned().collect();
    let rows: Vec<Row> = idx
        .iter()
        .map(|&i| {
            let f = &b.pop[i];
            Row {
                name: f.name(),
                kills: f.kills,
                dmg: f.dmg_dealt,
                fitness: f.fitness,
                advances: advance.contains(&i),
                wins: f.career_wins,
                career_kills: f.career_kills,
            }
        })
        .collect();
    let Battle { pop, events, t, .. } = b;
    let next: Vec<Fighter> = pop
        .into_iter()
        .enumerate()
        .filter(|(i, _)| advance.contains(i))
        .map(|(_, f)| f)
        .collect();
    (
        next,
        StageRecord {
            label: label.to_string(),
            from: n,
            keep,
            ticks: t,
            rows,
            events,
        },
    )
}

fn print_summary(records: &[StageRecord], completed: bool) {
    println!("\nHIVE BRAIN: tournament summary");
    for r in records {
        let kills = r.events.iter().filter(|e| e.killer.is_some()).count();
        let storm = r.events.len() - kills;
        println!(
            "\n== {}: {} fighters -> {} | {} ticks | {} blade kills, {} storm deaths ==",
            r.label, r.from, r.keep, r.ticks, kills, storm
        );
        let adv: Vec<&str> = r
            .rows
            .iter()
            .filter(|x| x.advances)
            .map(|x| x.name.as_str())
            .collect();
        let out: Vec<&str> = r
            .rows
            .iter()
            .filter(|x| !x.advances)
            .map(|x| x.name.as_str())
            .collect();
        println!("  advanced   : {}", adv.join(" "));
        if !out.is_empty() {
            println!("  eliminated : {}", out.join(" "));
        }
    }
    if completed {
        if let Some(c) = records.last().and_then(|r| r.rows.first()) {
            println!(
                "\n  CHAMPION: {}  ({} kills in the final, career {} wins / {} kills)",
                c.name, c.kills, c.wins, c.career_kills
            );
        }
    } else {
        println!("\n  (tournament stopped early)");
    }
    if let Ok(mut l) = std::fs::File::create("kills.log") {
        for r in records {
            for e in &r.events {
                writeln!(l, "{} {}", r.label, fmt_event(e)).ok();
            }
        }
        println!("  full kill log written to kills.log");
    }
}

// ---------- terminal UI ----------
struct Ui {
    stage: usize,
    label: &'static str,
    fighters: usize,
    keep: usize,
    speed: u32,
    paused: bool,
}

fn draw_battle(f: &mut Frame, b: &Battle, ui: &Ui) {
    let rows = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(12),
        Constraint::Length(10),
    ])
    .split(f.area());
    let mid = Layout::horizontal([Constraint::Min(30), Constraint::Length(42)]).split(rows[1]);

    // header
    let target = if ui.keep == 1 {
        "one champion".to_string()
    } else {
        format!("top {} advance", ui.keep)
    };
    let title = Line::from(vec![
        Span::styled(
            " HIVE BRAIN ",
            Style::new()
                .fg(Color::Black)
                .bg(Color::Red)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(
                "  {} ({}/{})  ·  {} fighters, {}  ·  tick {}  ·  alive {}  ·  storm radius {:.0}m",
                ui.label,
                ui.stage + 1,
                STAGES.len(),
                ui.fighters,
                target,
                b.t,
                b.alive(),
                b.storm_r
            ),
            Style::new().add_modifier(Modifier::BOLD),
        ),
    ]);
    let help = Line::styled(
        format!(
            " [space] {}   [+/-] speed x{}   [s] skip to end of fight   [q] quit",
            if ui.paused { "resume" } else { "pause " },
            ui.speed
        ),
        Style::new().fg(Color::DarkGray),
    );
    f.render_widget(
        Paragraph::new(vec![title, help]).block(Block::bordered()),
        rows[0],
    );

    // arena (kept round: terminal cells are about twice as tall as wide)
    let block = Block::bordered().title(" ARENA ");
    let inner = block.inner(mid[0]);
    f.render_widget(block, mid[0]);
    if inner.height > 2 {
        let h = inner.height;
        let w = (h * 2).min(inner.width);
        let rect = Rect {
            x: inner.x + inner.width.saturating_sub(w) / 2,
            y: inner.y,
            width: w,
            height: h,
        };
        let canvas = Canvas::default()
            .marker(Marker::Braille)
            .x_bounds([-74.0, 74.0])
            .y_bounds([-74.0, 74.0])
            .paint(|ctx| {
                ctx.draw(&Circle {
                    x: 0.0,
                    y: 0.0,
                    radius: ARENA_R as f64,
                    color: Color::DarkGray,
                });
                if b.t > STORM_START {
                    ctx.draw(&Circle {
                        x: 0.0,
                        y: 0.0,
                        radius: b.storm_r as f64,
                        color: Color::Red,
                    });
                }
                for (i, fi) in b.pop.iter().enumerate() {
                    if !fi.alive {
                        if fi.survived > 0 && fi.survived + 25 >= b.t {
                            ctx.print(
                                fi.x as f64,
                                fi.y as f64,
                                Span::styled("x", Style::new().fg(Color::DarkGray)),
                            );
                        }
                        continue;
                    }
                    let style = if b.attacked[i] {
                        Style::new()
                            .fg(Color::LightRed)
                            .add_modifier(Modifier::BOLD)
                    } else if b.was_hit[i] {
                        Style::new().fg(Color::Yellow)
                    } else if b.shielding[i] {
                        Style::new().fg(Color::Cyan)
                    } else {
                        Style::new().fg(Color::White)
                    };
                    ctx.print(
                        fi.x as f64,
                        fi.y as f64,
                        Span::styled((SYMS[i] as char).to_string(), style),
                    );
                }
            });
        f.render_widget(canvas, rect);
    }

    // roster
    let mut alive: Vec<usize> = (0..b.pop.len()).filter(|&i| b.pop[i].alive).collect();
    alive.sort_by(|&a, &c| b.pop[c].hp.partial_cmp(&b.pop[a].hp).unwrap());
    let items: Vec<ListItem> = alive
        .iter()
        .map(|&i| {
            let fi = &b.pop[i];
            let filled = ((fi.hp.max(0.0) / 100.0) * 8.0).ceil() as usize;
            let color = if fi.hp > 60.0 {
                Color::Green
            } else if fi.hp > 30.0 {
                Color::Yellow
            } else {
                Color::Red
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{} ", SYMS[i] as char),
                    Style::new().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!("{:<10} ", fi.name())),
                Span::styled(
                    format!(
                        "{}{}",
                        "█".repeat(filled.min(8)),
                        "░".repeat(8 - filled.min(8))
                    ),
                    Style::new().fg(color),
                ),
                Span::raw(format!(
                    " {:>3} {}K",
                    fi.hp.max(0.0).ceil() as i32,
                    fi.kills
                )),
            ]))
        })
        .collect();
    f.render_widget(
        List::new(items).block(Block::bordered().title(format!(
            " ALIVE {}/{} ",
            b.alive(),
            b.pop.len()
        ))),
        mid[1],
    );

    // kill feed
    let room = rows[2].height.saturating_sub(2) as usize;
    let from = b.events.len().saturating_sub(room);
    let feed: Vec<ListItem> = b.events[from..]
        .iter()
        .map(|e| {
            let style = if e.killer.is_some() {
                Style::new().fg(Color::LightRed)
            } else {
                Style::new().fg(Color::Yellow)
            };
            ListItem::new(Line::styled(fmt_event(e), style))
        })
        .collect();
    f.render_widget(
        List::new(feed)
            .block(Block::bordered().title(" KILL FEED  (red = blade, yellow = storm) ")),
        rows[2],
    );
}

fn draw_results(f: &mut Frame, rec: &StageRecord, last: bool) {
    let rows = Layout::vertical([Constraint::Min(8), Constraint::Length(3)]).split(f.area());
    let kills = rec.events.iter().filter(|e| e.killer.is_some()).count();
    let title = if last {
        format!(
            " {} RESULTS · {} fighters · {} ticks · {} kills ",
            rec.label, rec.from, rec.ticks, kills
        )
    } else {
        format!(
            " {} RESULTS · {} -> top {} · {} ticks · {} kills ",
            rec.label, rec.from, rec.keep, rec.ticks, kills
        )
    };
    let items: Vec<ListItem> = rec
        .rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let (tag, color) = if last && i == 0 {
                ("CHAMPION", Color::LightYellow)
            } else if r.advances {
                ("ADVANCES", Color::Green)
            } else {
                ("ELIMINATED", Color::Red)
            };
            ListItem::new(Line::styled(
                format!(
                    "{:>2}.  {:<10}  kills {:>2}   damage {:>5.0}   score {:>6.1}   {}",
                    i + 1,
                    r.name,
                    r.kills,
                    r.dmg,
                    r.fitness,
                    tag
                ),
                Style::new().fg(color),
            ))
        })
        .collect();
    f.render_widget(
        List::new(items).block(Block::bordered().title(title)),
        rows[0],
    );
    let hint = if last {
        "[enter] see the champion   [q] quit"
    } else {
        "[enter] next round   [q] quit"
    };
    f.render_widget(
        Paragraph::new(Line::styled(
            hint,
            Style::new().add_modifier(Modifier::BOLD),
        ))
        .alignment(Alignment::Center)
        .block(Block::bordered()),
        rows[1],
    );
}

fn draw_champion(f: &mut Frame, rec: &StageRecord) {
    let area = f.area();
    let champ = &rec.rows[0];
    let mut lines = vec![
        Line::from(""),
        Line::styled(
            "★  ★  ★   C H A M P I O N   ★  ★  ★",
            Style::new()
                .fg(Color::LightYellow)
                .add_modifier(Modifier::BOLD),
        ),
        Line::from(""),
        Line::styled(
            champ.name.clone(),
            Style::new()
                .fg(Color::LightYellow)
                .add_modifier(Modifier::BOLD),
        ),
        Line::from(""),
        Line::from(format!(
            "{} kills and {:.0} damage in the final",
            champ.kills, champ.dmg
        )),
        Line::from(format!(
            "career: {} wins, {} kills",
            champ.wins, champ.career_kills
        )),
        Line::from(""),
        Line::styled(
            "final standings",
            Style::new().add_modifier(Modifier::UNDERLINED),
        ),
    ];
    for (i, r) in rec.rows.iter().take(5).enumerate() {
        lines.push(Line::from(format!(
            "{}. {:<10}  {} kills",
            i + 1,
            r.name,
            r.kills
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::styled("[q] quit", Style::new().fg(Color::DarkGray)));
    let block = Block::bordered().title(" HIVE BRAIN ");
    f.render_widget(
        Paragraph::new(lines)
            .alignment(Alignment::Center)
            .block(block),
        area,
    );
}

fn is_quit(k: &KeyEvent) -> bool {
    matches!(k.code, KeyCode::Char('q') | KeyCode::Esc)
        || (k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL))
}

fn main() -> R<()> {
    let opts = parse_opts();
    let mut rng = StdRng::seed_from_u64(opts.seed);
    let interactive = std::io::stdout().is_terminal() && !opts.headless && !opts.snapshot;

    if opts.snapshot {
        return snapshot(&opts, &mut rng);
    }

    println!("HIVE BRAIN: {POP} candle networks, {GENOME_LEN} weights each, CPU only");
    let roster = train(&opts, &mut rng, std::io::stderr().is_terminal())?;

    if !interactive {
        let (records, done) = tournament_headless(roster, &mut rng)?;
        print_summary(&records, done);
        return Ok(());
    }

    let mut term = ratatui::init();
    let result = tournament_tui_run(&mut term, roster, &mut rng);
    ratatui::restore();
    let (records, done) = result?;
    print_summary(&records, done);
    Ok(())
}

fn tournament_headless(mut roster: Vec<Fighter>, rng: &mut StdRng) -> R<(Vec<StageRecord>, bool)> {
    let mut records = Vec::new();
    for (keep, label) in STAGES.iter() {
        let mut b = Battle::new(roster, rng);
        b.run_to_end(rng)?;
        let (next, rec) = conclude(b, *keep, label);
        roster = next;
        records.push(rec);
    }
    Ok((records, true))
}

fn tournament_tui_run(
    term: &mut ratatui::DefaultTerminal,
    roster: Vec<Fighter>,
    rng: &mut StdRng,
) -> R<(Vec<StageRecord>, bool)> {
    tournament_loop(term, roster, rng)
}

fn tournament_loop(
    term: &mut ratatui::DefaultTerminal,
    mut roster: Vec<Fighter>,
    rng: &mut StdRng,
) -> R<(Vec<StageRecord>, bool)> {
    let mut records: Vec<StageRecord> = Vec::new();
    let mut speed_idx = 2usize;
    for (si, (keep, label)) in STAGES.iter().enumerate() {
        let n = roster.len();
        let mut b = Battle::new(roster, rng);
        let mut paused = false;
        loop {
            let ui = Ui {
                stage: si,
                label,
                fighters: n,
                keep: *keep,
                speed: SPEEDS[speed_idx],
                paused,
            };
            term.draw(|f| draw_battle(f, &b, &ui))?;
            if b.done {
                break;
            }
            if event::poll(Duration::from_millis(33))? {
                if let CEvent::Key(k) = event::read()? {
                    if k.kind == KeyEventKind::Press {
                        if is_quit(&k) {
                            return Ok((records, false));
                        }
                        match k.code {
                            KeyCode::Char(' ') => paused = !paused,
                            KeyCode::Char('+') | KeyCode::Char('=') => {
                                speed_idx = (speed_idx + 1).min(SPEEDS.len() - 1)
                            }
                            KeyCode::Char('-') => speed_idx = speed_idx.saturating_sub(1),
                            KeyCode::Char('s') => b.run_to_end(rng)?,
                            _ => {}
                        }
                    }
                }
            }
            if !paused {
                for _ in 0..SPEEDS[speed_idx] {
                    b.step(rng)?;
                    if b.done {
                        break;
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(700));
        let (next, rec) = conclude(b, *keep, label);
        roster = next;
        let last = si + 1 == STAGES.len();
        records.push(rec);

        // results screen
        loop {
            let rec = records.last().unwrap();
            term.draw(|f| draw_results(f, rec, last))?;
            if let CEvent::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    if is_quit(&k) {
                        return Ok((records, false));
                    }
                    if matches!(k.code, KeyCode::Enter | KeyCode::Char(' ')) {
                        break;
                    }
                }
            }
        }
    }

    // champion screen
    loop {
        let rec = records.last().unwrap();
        term.draw(|f| draw_champion(f, rec))?;
        if let CEvent::Key(k) = event::read()? {
            if k.kind == KeyEventKind::Press && (is_quit(&k) || k.code == KeyCode::Enter) {
                break;
            }
        }
    }
    Ok((records, true))
}

// ---------- dev aid: render screens into plain text ----------
fn dump(term: &Terminal<TestBackend>) {
    let buf = term.backend().buffer().clone();
    for y in 0..buf.area.height {
        let mut line = String::new();
        for x in 0..buf.area.width {
            line.push_str(buf[(x, y)].symbol());
        }
        println!("{}", line.trim_end());
    }
}

fn snapshot(opts: &Opts, rng: &mut StdRng) -> R<()> {
    let roster = train(opts, rng, false)?;
    let mut b = Battle::new(roster, rng);
    for _ in 0..420 {
        b.step(rng)?;
    }
    let mut term = Terminal::new(TestBackend::new(130, 44))?;
    let ui = Ui {
        stage: 0,
        label: "ROUND 1",
        fighters: POP,
        keep: 25,
        speed: 4,
        paused: false,
    };
    term.draw(|f| draw_battle(f, &b, &ui))?;
    dump(&term);
    b.run_to_end(rng)?;
    let (_, rec) = conclude(b, 25, "ROUND 1");
    term.draw(|f| draw_results(f, &rec, false))?;
    dump(&term);
    term.draw(|f| draw_champion(f, &rec))?;
    dump(&term);
    Ok(())
}
