use std::collections::HashMap;

pub struct Cache {
    data: HashMap<(usize, usize), f64>,
}

impl Cache {
    pub fn new() -> Self {
        Cache {
            data: HashMap::new(),
        }
    }

    pub fn get(&self, i: usize, j: usize) -> Option<&f64> {
        self.data.get(&(i, j))
    }

    pub fn insert(&mut self, key: (usize, usize), value: f64) {
        self.data.insert(key, value);
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn get_or_insert(&mut self, key: (usize, usize), value: f64) -> &f64 {
        self.data.entry(key).or_insert(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache() {
        let mut cache = Cache::new();

        cache.insert((0, 0), 1.0);
        cache.insert((0, 1), 2.0);
        cache.insert((1, 0), 3.0);
        cache.insert((1, 1), 4.0);

        assert_eq!(cache.get(0, 0), Some(&1.0));
        assert_eq!(cache.get(0, 1), Some(&2.0));
        assert_eq!(cache.get(1, 0), Some(&3.0));
        assert_eq!(cache.get(1, 1), Some(&4.0));
        assert_eq!(cache.get(2, 2), None);
    }
}