#pragma once

#include <array>
#include <string>

struct Settings {
  const long int dim;
  const long int degree;

  const std::string mesh;
  const long int renumbering;
  const long int sol_id;
  const double sol_freq;
  const long int seed;
  const double max_diffusion;
  const long int num_bas_ref;
  const long int cycles;

  const std::array<double, 3> strong_threshold;
  const double toll;

  const bool output_results;
  const bool evaluate_errors;

  const bool make_view; // pooling
  const long int view_size; // grandezza matrice pooling M

  const std::string stats_filename; // file dove salviamo stat
  const std::string settings_filename; // 

  const bool output_setup_details;
};

Settings get_settings(const std::string& filename);