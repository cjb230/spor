#[derive(Clone)]
pub struct PythonRandom {
    state: [u32; 624],
    index: usize,
}

impl PythonRandom {
    pub fn new(seed: Option<i64>) -> Self {
        match seed {
            Some(seed) => Self::seeded(seed as i128),
            None => {
                let mut entropy = [0u8; 32];
                getrandom::fill(&mut entropy).expect("operating system random source unavailable");
                let words: Vec<u32> = entropy
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| u32::from_le_bytes(b))
                    .collect();
                Self::from_words(&words)
            }
        }
    }

    pub fn seeded(seed: i128) -> Self {
        let mut n = seed.unsigned_abs();
        let mut words = Vec::with_capacity(4);
        loop {
            words.push(n as u32);
            n >>= 32;
            if n == 0 {
                break;
            }
        }
        Self::from_words(&words)
    }

    pub fn from_words(key: &[u32]) -> Self {
        assert!(!key.is_empty());
        let mut r = Self {
            state: [0; 624],
            index: 624,
        };
        r.state[0] = 19650218;
        for i in 1..624 {
            r.state[i] = 1812433253u32
                .wrapping_mul(r.state[i - 1] ^ (r.state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        let (mut i, mut j) = (1, 0);
        for _ in 0..624.max(key.len()) {
            r.state[i] = (r.state[i]
                ^ (r.state[i - 1] ^ (r.state[i - 1] >> 30)).wrapping_mul(1664525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                r.state[0] = r.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            r.state[i] = (r.state[i]
                ^ (r.state[i - 1] ^ (r.state[i - 1] >> 30)).wrapping_mul(1566083941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                r.state[0] = r.state[623];
                i = 1;
            }
        }
        r.state[0] = 0x80000000;
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    pub fn bits(&mut self, k: u32) -> u64 {
        assert!((1..=64).contains(&k));
        if k <= 32 {
            (self.next_u32() >> (32 - k)) as u64
        } else {
            let low = self.next_u32() as u64;
            low | ((self.next_u32() >> (64 - k)) as u64) << 32
        }
    }

    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0);
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = self.bits(k) as usize;
            if r < n {
                return r;
            }
        }
    }

    pub fn shuffle<T>(&mut self, cards: &mut [T]) {
        for i in (1..cards.len()).rev() {
            let j = self.below(i + 1);
            cards.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn python_integer_seed_words() {
        for (seed, expected) in [
            (
                0,
                [3626764237, 1654615998, 3255389356, 3823568514, 1806341205],
            ),
            (
                1,
                [577090037, 2444712010, 3639700191, 3445702192, 3280387012],
            ),
            (
                4294967297,
                [991850117, 2151679444, 2200792185, 3616180723, 1780619055],
            ),
            (
                i64::MIN as i128,
                [2377109768, 1691753470, 4039908392, 899083848, 1039357674],
            ),
        ] {
            let mut rng = PythonRandom::seeded(seed);
            assert_eq!(std::array::from_fn::<_, 5, _>(|_| rng.next_u32()), expected);
        }
    }
    #[test]
    fn python_shuffle_and_choice_include_rejection_sampling() {
        let expected = [
            2, 38, 11, 25, 19, 9, 18, 49, 15, 29, 22, 5, 3, 53, 37, 32, 47, 50, 23, 43, 39, 21, 12,
            26, 33, 20, 52, 44, 35, 10, 34, 40, 14, 17, 45, 0, 27, 42, 1, 46, 6, 13, 24, 41, 30,
            28, 31, 7, 16, 4, 48, 51, 36, 8,
        ];
        for seed in [1, -1] {
            let mut deck: Vec<u8> = (0..54).collect();
            PythonRandom::seeded(seed).shuffle(&mut deck);
            assert_eq!(deck, expected);
        }
        let mut rng = PythonRandom::seeded(3);
        assert_eq!(
            [1, 2, 54, 3, 9, 1, 100].map(|n| rng.below(n)),
            [0, 0, 23, 2, 7, 0, 77]
        );
    }
}
