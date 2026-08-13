// ms_hnsw.cpp  (v3 - no neighbor pruning, with brute-force baseline mode)
//
// Integrated speedups:
//   1) Timestamp visited array in searchLayer() (no O(N) clear per call)
//   2) Replace unordered_set in score_best_matches() with flat arrays
//   3) Precompute per-spectrum norm for default (mz_power=0, intensity_power=1) and skip pow() in that path
//   4) Add reserve() and addSpectrum(Spectrum&&) to remove copies
//
// Modes:
//   build   - Build HNSW index from MGF library
//   query   - Query HNSW index
//   baseline - Brute-force k-NN for ground truth
//
// Usage:
//   ./ms_hnsw build LIB.mgf INDEX.bin M efConstruction
//   ./ms_hnsw query INDEX.bin QUERIES.mgf K efSearch
//   ./ms_hnsw baseline LIB.mgf QUERIES.mgf K

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <string>
#include <random>
#include <queue>
#include <limits>
#include <cstdint>

using namespace std;

// ---------------------------------------------------------
// Spectrum and spectral similarity
// ---------------------------------------------------------

struct Spectrum {
    vector<double> mz;
    vector<double> intensity;
    string title;
    string pepmass;

    // Precomputed for default similarity path: mz_power=0.0, intensity_power=1.0
    // norm = sum_i (intensity_i^2)
    double norm = 0.0;
};

static inline double compute_default_norm(const vector<double>& intensity) {
    double s = 0.0;
    for (double x : intensity) s += x * x;
    return s;
}

tuple<double, int> score_best_matches(
    const vector<tuple<int, int, double>>& matching_pairs,
    const vector<double>& spec1_mz, const vector<double>& spec1_intensity,
    const vector<double>& spec2_mz, const vector<double>& spec2_intensity,
    double mz_power = 0.0, double intensity_power = 1.0,
    // fast path: if provided, use these instead of recomputing per-call
    const double* spec1_norm_default = nullptr,
    const double* spec2_norm_default = nullptr,
    bool use_default_fast_path = false)
{
    double score = 0.0;
    int used_matches = 0;

    // Flat arrays instead of unordered_set
    vector<uint8_t> used1(spec1_mz.size(), 0);
    vector<uint8_t> used2(spec2_mz.size(), 0);

    for (const auto& mp : matching_pairs) {
        int idx1 = get<0>(mp);
        int idx2 = get<1>(mp);
        if (!used1[idx1] && !used2[idx2]) {
            used1[idx1] = 1;
            used2[idx2] = 1;
            score += get<2>(mp);
            used_matches++;
        }
    }

    double spec1_power = 0.0;
    double spec2_power = 0.0;

    if (use_default_fast_path && spec1_norm_default && spec2_norm_default) {
        spec1_power = *spec1_norm_default;
        spec2_power = *spec2_norm_default;
    } else {
        // General (slower) path
        for (size_t i = 0; i < spec1_mz.size(); ++i) {
            double v = pow(spec1_mz[i], mz_power) * pow(spec1_intensity[i], intensity_power);
            spec1_power += v * v;
        }
        for (size_t i = 0; i < spec2_mz.size(); ++i) {
            double v = pow(spec2_mz[i], mz_power) * pow(spec2_intensity[i], intensity_power);
            spec2_power += v * v;
        }
    }

    if (spec1_power == 0.0 || spec2_power == 0.0) return make_tuple(0.0, used_matches);

    score /= (sqrt(spec1_power) * sqrt(spec2_power));
    return make_tuple(score, used_matches);
}

vector<pair<int,int>> find_matches(
    const vector<double>& spec1_mz, const vector<double>& spec2_mz,
    double tolerance, double shift = 0.0)
{
    int lowest_idx = 0;
    vector<pair<int,int>> matches;

    for (size_t peak1_idx = 0; peak1_idx < spec1_mz.size(); ++peak1_idx) {
        double mz = spec1_mz[peak1_idx];
        double low_bound = mz - tolerance;
        double high_bound = mz + tolerance;

        for (size_t peak2_idx = lowest_idx; peak2_idx < spec2_mz.size(); ++peak2_idx) {
            double mz2 = spec2_mz[peak2_idx] + shift;
            if (mz2 > high_bound) break;
            if (mz2 < low_bound) {
                lowest_idx = static_cast<int>(peak2_idx + 1);
            } else {
                matches.emplace_back((int)peak1_idx, (int)peak2_idx);
            }
        }
    }
    return matches;
}

double spectral_similarity(
    const Spectrum& s1, const Spectrum& s2,
    double mz_power = 0.0, double intensity_power = 1.0,
    double tolerance = 0.01, double shift = 0.0)
{
    if (s1.mz.empty() || s2.mz.empty()) return 0.0;

    auto matches = find_matches(s1.mz, s2.mz, tolerance, shift);
    if (matches.empty()) return 0.0;

    vector<tuple<int, int, double>> matching_pairs;
    matching_pairs.reserve(matches.size());

    const bool default_fast = (mz_power == 0.0 && intensity_power == 1.0);

    if (default_fast) {
        // weight = intensity1 * intensity2 (no pow)
        for (const auto& m : matches) {
            int i1 = m.first;
            int i2 = m.second;
            double w1 = s1.intensity[i1];
            double w2 = s2.intensity[i2];
            matching_pairs.emplace_back(i1, i2, w1 * w2);
        }
    } else {
        // General (slower) path
        for (const auto& m : matches) {
            int i1 = m.first;
            int i2 = m.second;
            double power1 = pow(s1.mz[i1], mz_power) * pow(s1.intensity[i1], intensity_power);
            double power2 = pow(s2.mz[i2], mz_power) * pow(s2.intensity[i2], intensity_power);
            matching_pairs.emplace_back(i1, i2, power1 * power2);
        }
    }

    sort(matching_pairs.begin(), matching_pairs.end(),
         [](const auto& a, const auto& b) {
            return get<2>(a) > get<2>(b);
         });

    auto result = score_best_matches(
        matching_pairs, s1.mz, s1.intensity, s2.mz, s2.intensity,
        mz_power, intensity_power,
        /*spec1_norm_default=*/ default_fast ? &s1.norm : nullptr,
        /*spec2_norm_default=*/ default_fast ? &s2.norm : nullptr,
        /*use_default_fast_path=*/ default_fast);

    double score = get<0>(result);
    return max(0.0, min(1.0, score));
}

// ---------------------------------------------------------
// Global distance counter
// ---------------------------------------------------------

long long g_dist_calls = 0;

inline double spectral_distance(const Spectrum& a, const Spectrum& b) {
    ++g_dist_calls;
    double sim = spectral_similarity(a, b);
    return 1.0 - sim;
}

// ---------------------------------------------------------
// HNSW structures
// ---------------------------------------------------------

struct HNSWNode {
    int level;
    vector<vector<int>> neighbors;
};

struct Candidate {
    int id;
    double dist;
};

struct CompareMin {
    bool operator()(const Candidate& a, const Candidate& b) const {
        return a.dist > b.dist;
    }
};

struct CompareMax {
    bool operator()(const Candidate& a, const Candidate& b) const {
        return a.dist < b.dist;
    }
};

class HNSWIndex {
public:
    HNSWIndex(int M = 16, int efConstruction = 200)
        : M_(M),
          efConstruction_(efConstruction),
          enterPoint_(-1),
          maxLevel_(-1),
          rng_(std::random_device{}()),
          uni_(0.0, 1.0)
    {
        mL_ = 1.0 / std::log((double)M_);
    }

    // Reserve memory upfront (reduces realloc/copies during build)
    void reserve(size_t n) {
        nodes_.reserve(n);
        spectra_.reserve(n);
        visited_tag_.reserve(n);
    }

    // Copying overload (kept for convenience)
    int addSpectrum(const Spectrum& s) {
        Spectrum tmp = s;
        return addSpectrum(std::move(tmp));
    }

    // Move overload to remove copies during build
    int addSpectrum(Spectrum&& s) {
        // Ensure norm is set (in case caller didn't)
        if (s.norm == 0.0 && !s.intensity.empty()) {
            // If truly zero norm (all intensities 0) this stays 0.
            // If norm wasn't computed but should be, compute it here.
            // This is safe; worst case it recomputes once.
            s.norm = compute_default_norm(s.intensity);
        }

        int newId = (int)spectra_.size();
        spectra_.push_back(std::move(s));

        HNSWNode node;
        node.level = sampleLevel();
        node.neighbors.assign(node.level + 1, {});

        if (nodes_.empty()) {
            nodes_.push_back(std::move(node));
            enterPoint_ = newId;
            maxLevel_ = nodes_[newId].level;
            return newId;
        }

        nodes_.push_back(std::move(node));

        int cur = enterPoint_;
        const Spectrum& qspec = spectra_[newId];

        // Greedy search from top level down to node's level + 1
        for (int level = maxLevel_; level > nodes_[newId].level; --level) {
            vector<Candidate> cand = searchLayer(qspec, cur, 1, level, nullptr);
            cur = closestFromList(cand);
        }

        // Insert into levels from min(newLevel, maxLevel) down to 0
        int topLevel = std::min(nodes_[newId].level, maxLevel_);
        for (int level = topLevel; level >= 0; --level) {
            vector<Candidate> candidates = searchLayer(qspec, cur, efConstruction_, level, nullptr);

            // Select M closest neighbors (no pruning of existing nodes)
            vector<int> neigh = selectNeighborsSimple(candidates, M_);

            // Add bidirectional edges (no pruning - allow unlimited neighbors)
            for (int nid : neigh) {
                nodes_[newId].neighbors[level].push_back(nid);
                nodes_[nid].neighbors[level].push_back(newId);
            }

            cur = closestFromList(candidates);
        }

        if (nodes_[newId].level > maxLevel_) {
            maxLevel_ = nodes_[newId].level;
            enterPoint_ = newId;
        }

        return newId;
    }

    vector<int> searchKnn(const Spectrum& qspec, int K, int efSearch,
                          size_t* visited_out = nullptr) const
    {
        vector<int> result;
        if (nodes_.empty()) return result;

        int cur = enterPoint_;
        size_t visited_total = 0;

        // Greedy search in upper layers
        for (int level = maxLevel_; level > 0; --level) {
            vector<Candidate> cand = searchLayer(qspec, cur, 1, level, &visited_total);
            cur = closestFromList(cand);
        }

        // Full search in layer 0
        vector<Candidate> candidates = searchLayer(qspec, cur, efSearch, 0, &visited_total);
        if (visited_out) *visited_out = visited_total;

        // Sort by distance and return top K
        sort(candidates.begin(), candidates.end(),
             [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });

        int n = std::min(K, (int)candidates.size());
        result.reserve(n);
        for (int i = 0; i < n; ++i) {
            result.push_back(candidates[i].id);
        }
        return result;
    }

    const Spectrum& getSpectrum(int id) const { return spectra_[id]; }
    int size() const { return (int)spectra_.size(); }

    bool save(const string& path) const {
        ofstream out(path, ios::binary);
        if (!out) return false;

        const char magic[8] = {'H','N','S','W','M','S','3','\0'};  // Version 3
        out.write(magic, 8);
        uint32_t version = 3;
        out.write((char*)&version, sizeof(version));

        int32_t M = M_;
        int32_t efC = efConstruction_;
        int32_t n = (int32_t)nodes_.size();
        int32_t maxL = maxLevel_;
        int32_t ep = enterPoint_;

        out.write((char*)&M, sizeof(M));
        out.write((char*)&efC, sizeof(efC));
        out.write((char*)&n, sizeof(n));
        out.write((char*)&maxL, sizeof(maxL));
        out.write((char*)&ep, sizeof(ep));

        for (int i = 0; i < n; ++i) {
            int32_t level = nodes_[i].level;
            out.write((char*)&level, sizeof(level));
            for (int l = 0; l <= level; ++l) {
                int32_t deg = (int32_t)nodes_[i].neighbors[l].size();
                out.write((char*)&deg, sizeof(deg));
                out.write((char*)nodes_[i].neighbors[l].data(),
                          deg * sizeof(int32_t));
            }
        }

        for (int i = 0; i < n; ++i) {
            const auto& s = spectra_[i];
            int32_t len = (int32_t)s.mz.size();
            out.write((char*)&len, sizeof(len));
            if (len > 0) {
                out.write((char*)s.mz.data(), len * sizeof(double));
                out.write((char*)s.intensity.data(), len * sizeof(double));
            }
            int32_t pep_len = (int32_t)s.pepmass.size();
            out.write((char*)&pep_len, sizeof(pep_len));
            if (pep_len > 0) {
                out.write(s.pepmass.data(), pep_len);
            }
        }
        return true;
    }

    bool load(const string& path) {
        ifstream in(path, ios::binary);
        if (!in) return false;

        char magic[8];
        in.read(magic, 8);

        uint32_t version;
        in.read((char*)&version, sizeof(version));

        int32_t M, efC, n, maxL, ep;

        string magic_str(magic, magic + 7);
        if (magic_str == "HNSWMS1" || magic_str == "HNSWMS3") {
            // Version 1 or 3 format
            in.read((char*)&M, sizeof(M));
            in.read((char*)&efC, sizeof(efC));
            in.read((char*)&n, sizeof(n));
            in.read((char*)&maxL, sizeof(maxL));
            in.read((char*)&ep, sizeof(ep));
        } else if (magic_str == "HNSWMS2") {
            // Version 2 format (has Mmax0)
            int32_t Mmax0;
            in.read((char*)&M, sizeof(M));
            in.read((char*)&Mmax0, sizeof(Mmax0));
            in.read((char*)&efC, sizeof(efC));
            in.read((char*)&n, sizeof(n));
            in.read((char*)&maxL, sizeof(maxL));
            in.read((char*)&ep, sizeof(ep));
        } else {
            cerr << "Invalid index file magic\n";
            return false;
        }

        M_ = M;
        efConstruction_ = efC;
        maxLevel_ = maxL;
        enterPoint_ = ep;
        mL_ = 1.0 / std::log((double)M_);

        nodes_.assign(n, HNSWNode());
        spectra_.assign(n, Spectrum());

        for (int i = 0; i < n; ++i) {
            int32_t level;
            in.read((char*)&level, sizeof(level));
            nodes_[i].level = level;
            nodes_[i].neighbors.assign(level + 1, {});
            for (int l = 0; l <= level; ++l) {
                int32_t deg;
                in.read((char*)&deg, sizeof(deg));
                nodes_[i].neighbors[l].resize(deg);
                in.read((char*)nodes_[i].neighbors[l].data(),
                        deg * sizeof(int32_t));
            }
        }

        for (int i = 0; i < n; ++i) {
            Spectrum& s = spectra_[i];
            int32_t len;
            in.read((char*)&len, sizeof(len));
            s.mz.resize(len);
            s.intensity.resize(len);
            if (len > 0) {
                in.read((char*)s.mz.data(), len * sizeof(double));
                in.read((char*)s.intensity.data(), len * sizeof(double));
            }
            int32_t pep_len;
            in.read((char*)&pep_len, sizeof(pep_len));
            if (pep_len > 0) {
                s.pepmass.resize(pep_len);
                in.read(&s.pepmass[0], pep_len);
            }

            // Recompute default norm (not stored in file to preserve v3 format)
            s.norm = compute_default_norm(s.intensity);
        }

        // Ensure visited array sized for searches
        visited_tag_.clear();
        visited_tag_.resize(nodes_.size(), 0);
        visit_token_ = 1;

        return true;
    }

    void printGraphStats() const {
        size_t n = nodes_.size();
        size_t total_deg = 0;
        size_t edges = 0;
        size_t max_deg = 0;
        vector<size_t> level_hist(maxLevel_ + 1, 0);

        for (size_t i = 0; i < n; ++i) {
            int lvl = nodes_[i].level;
            if (lvl >= 0 && lvl < (int)level_hist.size()) level_hist[lvl]++;
            for (int l = 0; l <= nodes_[i].level; ++l) {
                size_t deg = nodes_[i].neighbors[l].size();
                total_deg += deg;
                edges += deg;
                if (deg > max_deg) max_deg = deg;
            }
        }
        double avg_deg = n ? (double)total_deg / (double)n : 0.0;

        cerr << "GRAPH nodes=" << n
             << " out_edges=" << edges
             << " avg_out_degree=" << avg_deg
             << " max_degree=" << max_deg
             << " max_level=" << maxLevel_ << "\n";

        cerr << "GRAPH level_hist:";
        for (int l = 0; l <= maxLevel_; ++l) cerr << " L" << l << "=" << level_hist[l];
        cerr << "\n";
    }

private:
    int M_;
    int efConstruction_;
    double mL_;

    vector<HNSWNode> nodes_;
    vector<Spectrum> spectra_;
    int enterPoint_;
    int maxLevel_;

    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> uni_;

    // Timestamp visited array (no per-call O(N) clear)
    mutable vector<uint32_t> visited_tag_;
    mutable uint32_t visit_token_ = 1;

    int sampleLevel() {
        double u = uni_(rng_);
        return (int)std::floor(-std::log(u) * mL_);
    }

    inline void ensureVisitedSize() const {
        if (visited_tag_.size() < nodes_.size()) {
            visited_tag_.resize(nodes_.size(), 0);
        }
    }

    // Returns candidates with pre-computed distances
    vector<Candidate> searchLayer(const Spectrum& qspec, int entryId, int ef, int level,
                                  size_t* visited_out) const
    {
        priority_queue<Candidate, vector<Candidate>, CompareMin> C;
        priority_queue<Candidate, vector<Candidate>, CompareMax> W;

        ensureVisitedSize();

        uint32_t my_token = visit_token_++;
        if (visit_token_ == 0) {
            // overflow: reset tags
            std::fill(visited_tag_.begin(), visited_tag_.end(), 0);
            visit_token_ = 1;
            my_token = visit_token_++;
        }

        auto is_visited = [&](int id) -> bool { return visited_tag_[id] == my_token; };
        auto mark_visited = [&](int id) { visited_tag_[id] = my_token; };

        size_t local_visited = 0;

        double dist_entry = spectral_distance(spectra_[entryId], qspec);
        C.push({entryId, dist_entry});
        W.push({entryId, dist_entry});
        mark_visited(entryId);
        local_visited++;

        while (!C.empty()) {
            Candidate curr = C.top();
            C.pop();

            double worstDist = W.top().dist;
            if (curr.dist > worstDist) break;

            const HNSWNode& node = nodes_[curr.id];
            if (level > node.level) continue;

            for (int nid : node.neighbors[level]) {
                if (is_visited(nid)) continue;
                mark_visited(nid);
                local_visited++;

                double d = spectral_distance(spectra_[nid], qspec);
                if ((int)W.size() < ef || d < W.top().dist) {
                    C.push({nid, d});
                    W.push({nid, d});
                    if ((int)W.size() > ef) W.pop();
                }
            }
        }

        if (visited_out) *visited_out += local_visited;

        vector<Candidate> res;
        res.reserve(W.size());
        while (!W.empty()) {
            res.push_back(W.top());
            W.pop();
        }

        // Sort by distance ascending
        sort(res.begin(), res.end(), [](const Candidate& a, const Candidate& b){
            return a.dist < b.dist;
        });

        return res;
    }

    // Uses pre-computed distances from candidates
    int closestFromList(const vector<Candidate>& cand) const {
        if (cand.empty()) return enterPoint_;
        return cand.front().id;  // Already sorted by distance
    }

    // Select M closest neighbors from pre-sorted candidates
    vector<int> selectNeighborsSimple(const vector<Candidate>& candidates, int M) const {
        int n = std::min(M, (int)candidates.size());
        vector<int> res;
        res.reserve(n);
        for (int i = 0; i < n; ++i) res.push_back(candidates[i].id);
        return res;
    }
};

// ---------------------------------------------------------
// Brute-force baseline
// ---------------------------------------------------------

vector<vector<int>> bruteforce_knn(
    const vector<Spectrum>& library,
    const vector<Spectrum>& queries,
    int K)
{
    int Q = (int)queries.size();
    int N = (int)library.size();
    vector<vector<int>> results(Q);

    cerr << "Brute-force k-NN: " << Q << " queries against " << N << " library spectra\n";

    auto start = std::chrono::high_resolution_clock::now();

    for (int qi = 0; qi < Q; ++qi) {
        if ((qi + 1) % 10 == 0 || qi == Q - 1) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double eta = (qi > 0) ? elapsed / qi * (Q - qi) : 0.0;
            cerr << "[baseline] query " << (qi + 1) << "/" << Q
                 << " elapsed=" << elapsed << "s ETA=" << eta << "s\n";
        }

        vector<pair<double, int>> sims;
        sims.reserve(N);

        for (int i = 0; i < N; ++i) {
            double sim = spectral_similarity(queries[qi], library[i]);
            sims.emplace_back(sim, i);
        }

        // Sort by similarity descending
        partial_sort(sims.begin(), sims.begin() + min(K, N), sims.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

        results[qi].reserve(K);
        for (int i = 0; i < min(K, N); ++i) {
            results[qi].push_back(sims[i].second);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end - start).count();
    cerr << "Brute-force total time: " << total_sec << " s\n";
    cerr << "Brute-force avg per query: " << (total_sec / Q * 1000) << " ms\n";

    return results;
}

// ---------------------------------------------------------
// MGF parsing
// ---------------------------------------------------------

vector<Spectrum> read_mgf(const string& path, int max_spectra = -1) {
    ifstream file(path);
    vector<Spectrum> spectra;
    if (!file) {
        cerr << "Cannot open MGF file: " << path << "\n";
        return spectra;
    }

    string line;
    Spectrum current;
    bool in_ions = false;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (line.rfind("BEGIN IONS", 0) == 0) {
            in_ions = true;
            current = Spectrum();
        } else if (line.rfind("END IONS", 0) == 0) {
            if (!current.mz.empty()) {
                vector<pair<double,double>> tmp;
                tmp.reserve(current.mz.size());
                for (size_t i = 0; i < current.mz.size(); ++i)
                    tmp.emplace_back(current.mz[i], current.intensity[i]);
                sort(tmp.begin(), tmp.end(),
                     [](auto& a, auto& b){ return a.first < b.first; });

                current.mz.clear();
                current.intensity.clear();
                current.mz.reserve(tmp.size());
                current.intensity.reserve(tmp.size());
                for (auto& p : tmp) {
                    current.mz.push_back(p.first);
                    current.intensity.push_back(p.second);
                }
            }

            // Precompute default norm once
            current.norm = compute_default_norm(current.intensity);

            spectra.push_back(std::move(current));
            in_ions = false;

            if (max_spectra > 0 && (int)spectra.size() >= max_spectra) break;
        } else if (!in_ions) {
            continue;
        } else if (line.rfind("PEPMASS=", 0) == 0) {
            current.pepmass = line.substr(8);
        } else if (line.rfind("TITLE=", 0) == 0) {
            current.title = line.substr(6);
        } else if (std::isdigit((unsigned char)line[0])) {
            std::istringstream iss(line);
            double m, inten;
            if (iss >> m >> inten) {
                current.mz.push_back(m);
                current.intensity.push_back(inten);
            }
        }
    }

    cerr << "Loaded " << spectra.size() << " spectra from " << path << "\n";
    return spectra;
}

// ---------------------------------------------------------
// Main CLI
// ---------------------------------------------------------

void print_usage(const char* prog) {
    cerr << "Usage:\n"
         << "  " << prog << " build LIB.mgf INDEX.bin M efConstruction\n"
         << "  " << prog << " query INDEX.bin QUERIES.mgf K efSearch\n"
         << "  " << prog << " baseline LIB.mgf QUERIES.mgf K\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    string mode = argv[1];

    if (mode == "build") {
        if (argc < 6) {
            cerr << "build: need LIB.mgf INDEX.bin M efConstruction\n";
            return 1;
        }
        string lib_path = argv[2];
        string index_path = argv[3];
        int M = std::stoi(argv[4]);
        int efC = std::stoi(argv[5]);

        auto spectra = read_mgf(lib_path);
        if (spectra.empty()) {
            cerr << "No spectra loaded, abort.\n";
            return 1;
        }

        HNSWIndex index(M, efC);
        index.reserve(spectra.size()); // <--- new

        int n = (int)spectra.size();
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n; ++i) {
            if (i % 1000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start).count();
                double rate = (i > 0 ? i / elapsed : 0.0);
                double eta = (i > 0 ? (n - i) / rate : 0.0);
                cerr << "[build] " << i << "/" << n
                     << " (" << 100.0 * i / n << "%)"
                     << " elapsed=" << elapsed << " s"
                     << " ETA=" << eta / 60.0 << " min\n";
            }
            // Move spectra into index to avoid copying large vectors
            index.addSpectrum(std::move(spectra[i])); // <--- new
        }

        auto end = std::chrono::high_resolution_clock::now();
        double build_sec = std::chrono::duration<double>(end - start).count();
        cerr << "Index built for " << index.size() << " spectra.\n";
        cerr << "Build time: " << build_sec << " s\n";

        index.printGraphStats();

        if (!index.save(index_path)) {
            cerr << "Failed to save index to " << index_path << "\n";
            return 1;
        }
        cerr << "Index saved to " << index_path << "\n";
        return 0;
    }
    else if (mode == "query") {
        if (argc < 6) {
            cerr << "query: need INDEX.bin QUERIES.mgf K efSearch\n";
            return 1;
        }
        string index_path = argv[2];
        string queries_path = argv[3];
        int K = std::stoi(argv[4]);
        int efSearch = std::stoi(argv[5]);

        HNSWIndex index;
        if (!index.load(index_path)) {
            cerr << "Failed to load index from " << index_path << "\n";
            return 1;
        }

        auto queries = read_mgf(queries_path);
        if (queries.empty()) {
            cerr << "No queries loaded.\n";
            return 1;
        }

        int Q = (int)queries.size();
        cerr << "Running " << Q << " queries, K=" << K
             << " efSearch=" << efSearch << "\n";

        vector<vector<int>> hnsw_results(Q);

        double total_ms = 0.0;
        long long dist_sum = 0;
        size_t visited_sum = 0;

        for (int i = 0; i < Q; ++i) {
            size_t visited_count = 0;
            long long before = g_dist_calls;

            auto t0 = std::chrono::high_resolution_clock::now();
            hnsw_results[i] = index.searchKnn(queries[i], K, efSearch, &visited_count);
            auto t1 = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
            total_ms += ms;

            long long after = g_dist_calls;
            long long dist_calls = after - before;
            dist_sum += dist_calls;
            visited_sum += visited_count;
        }

        double avg_ms = total_ms / Q;
        double avg_dist_calls = (double)dist_sum / (double)Q;
        double avg_visited = (double)visited_sum / (double)Q;

        cerr << "HNSW total query time: " << total_ms << " ms\n";
        cerr << "HNSW avg per query: " << avg_ms << " ms\n";
        cerr << "STATS avg_dist_calls=" << avg_dist_calls
             << " avg_visited=" << avg_visited << "\n";

        for (int i = 0; i < Q; ++i) {
            for (int j = 0; j < (int)hnsw_results[i].size(); ++j) {
                if (j) cout << ' ';
                cout << hnsw_results[i][j];
            }
            cout << "\n";
        }

        return 0;
    }
    else if (mode == "baseline") {
        if (argc < 5) {
            cerr << "baseline: need LIB.mgf QUERIES.mgf K\n";
            return 1;
        }
        string lib_path = argv[2];
        string queries_path = argv[3];
        int K = std::stoi(argv[4]);

        auto library = read_mgf(lib_path);
        if (library.empty()) {
            cerr << "No library spectra loaded.\n";
            return 1;
        }

        auto queries = read_mgf(queries_path);
        if (queries.empty()) {
            cerr << "No queries loaded.\n";
            return 1;
        }

        auto results = bruteforce_knn(library, queries, K);

        // Output results to stdout (same format as query mode)
        for (int i = 0; i < (int)results.size(); ++i) {
            for (int j = 0; j < (int)results[i].size(); ++j) {
                if (j) cout << ' ';
                cout << results[i][j];
            }
            cout << "\n";
        }

        return 0;
    }
    else {
        cerr << "Unknown mode: " << mode << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
