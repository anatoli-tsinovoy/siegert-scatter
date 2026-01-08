%% SPS Test Suite - Concrete outputs for Python comparison
% This script exercises all functions in the SPS codebase and outputs
% numerical results that can be compared with the Python implementation.
%
% Run with: octave SPS_test_suite.m > test_outputs.txt

format long e;
disp('================================================================');
disp('SPS TEST SUITE - Reference outputs for Python implementation');
disp('================================================================');
disp(' ');

%% ============================================================
%% TEST 1: Jacobi Polynomials (j_polynomial.m)
%% ============================================================
disp('=== TEST 1: Jacobi Polynomials ===');
disp(' ');

% Test case 1.1: Simple evaluation at specific points
x_test = [-1; -0.5; 0; 0.5; 1];
alpha = 0;
beta = 4;  % Common case in SPS (beta = 2*l for l=2)
n_max = 5;

JP = j_polynomial(length(x_test), n_max, alpha, beta, x_test);
disp('Test 1.1: j_polynomial(5, 5, 0, 4, [-1,-0.5,0,0.5,1])');
disp('Shape: [5 x 6]');
disp('Values:');
disp(JP);

% Test case 1.2: Used in TISE for l=0
alpha_l0 = 0;
beta_l0 = 0;  % 2*l = 0
x_l0 = linspace(-0.9, 1, 5).';
JP_l0 = j_polynomial(length(x_l0), 3, alpha_l0, beta_l0, x_l0);
disp('Test 1.2: j_polynomial for l=0 case (alpha=0, beta=0)');
disp('x values:');
disp(x_l0.');
disp('JP(x):');
disp(JP_l0);

%% ============================================================
%% TEST 2: Gaussian Quadrature (get_gaussian_quadrature.m)
%% ============================================================
disp(' ');
disp('=== TEST 2: Gaussian Quadrature ===');
disp(' ');

% Test case 2.1: Small order, l=0 case
N_test = 5;
l_test = 0;
[omega_i, x_i] = get_gaussian_quadrature(N_test, 0, 2*l_test, -1, 1);
disp(['Test 2.1: Gauss-Jacobi quadrature N=', num2str(N_test), ', alpha=0, beta=', num2str(2*l_test)]);
disp('Nodes (x_i):');
disp(x_i.');
disp('Weights (omega_i):');
disp(omega_i.');

% Test case 2.2: l=2 case (used frequently)
l_test = 2;
[omega_i_l2, x_i_l2] = get_gaussian_quadrature(N_test, 0, 2*l_test, -1, 1);
disp(['Test 2.2: Gauss-Jacobi quadrature N=', num2str(N_test), ', alpha=0, beta=', num2str(2*l_test)]);
disp('Nodes (x_i):');
disp(x_i_l2.');
disp('Weights (omega_i):');
disp(omega_i_l2.');

% Test case 2.3: Larger N (typical in actual calculations)
N_test = 20;
l_test = 0;
[omega_i_20, x_i_20] = get_gaussian_quadrature(N_test, 0, 2*l_test, -1, 1);
disp(['Test 2.3: Gauss-Jacobi quadrature N=', num2str(N_test), ', alpha=0, beta=0']);
disp('First 5 nodes (x_i):');
disp(x_i_20(1:5).');
disp('Last 5 nodes (x_i):');
disp(x_i_20(end-4:end).');
disp('First 5 weights (omega_i):');
disp(omega_i_20(1:5).');

%% ============================================================
%% TEST 3: Spherical Bessel zeros (calc_z_l.m)
%% ============================================================
disp(' ');
disp('=== TEST 3: Spherical Bessel Zeros ===');
disp(' ');

for l = 0:5
    z_l = calc_z_l(l, false);
    disp(['Test 3.', num2str(l+1), ': calc_z_l(', num2str(l), ', false)']);
    disp(['Number of zeros: ', num2str(length(z_l))]);
    if ~isempty(z_l)
        disp('Values:');
        for i = 1:length(z_l)
            disp(['  z_', num2str(l), '_', num2str(i), ' = ', num2str(real(z_l(i)), '%.15e'), ...
                  ' + ', num2str(imag(z_l(i)), '%.15e'), 'i']);
        end
    else
        disp('  (empty for l=0)');
    end
end

%% ============================================================
%% TEST 4: medfilt1 (median filter)
%% ============================================================
disp(' ');
disp('=== TEST 4: Median Filter ===');
disp(' ');

x_med = [1, 5, 2, 8, 3, 7, 4, 6, 9, 0];
y_med3 = medfilt1(x_med, 3);
y_med5 = medfilt1(x_med, 5);
disp('Test 4.1: medfilt1([1,5,2,8,3,7,4,6,9,0], 3)');
disp('Input:');
disp(x_med);
disp('Output (window=3):');
disp(y_med3);
disp('Test 4.2: medfilt1(..., 5)');
disp('Output (window=5):');
disp(y_med5);

%% ============================================================
%% TEST 5: TISE_by_SPS (core eigenvalue solver)
%% ============================================================
disp(' ');
disp('=== TEST 5: TISE Eigenvalue Solver ===');
disp(' ');

% Test potential: Pöschl-Teller with lambda=4
% V(x) = -lambda*(lambda+1)/2 / cosh^2(x) = -10/cosh^2(x) for lambda=4
% Full-line bound states: E_n = -(lambda - n)^2 / 2 for n = 0,1,2,3
%   E_0 = -8 (even), E_1 = -4.5 (odd), E_2 = -2 (even), E_3 = -0.5 (odd)
% For RADIAL l=0 with psi(0)=0: only ODD parity states survive -> E = -4.5, -0.5
a0 = 1;
HBAR = 1;
MU = 1;
LAMBDA_PT = 4;
f_test = @(x) (-LAMBDA_PT*(LAMBDA_PT+1)/2./cosh(x).^2) * (MU*a0^2/HBAR^2);

% Test case 5.1: Small N, l=0
N_tise = 10;
a_tise = 10;
l_tise = 0;
[k_n_10, c_ctilde_zeta, eigenmodes, r_i] = TISE_by_SPS(f_test, N_tise, a_tise, l_tise);
disp(['Test 5.1: TISE_by_SPS with N=', num2str(N_tise), ', a=', num2str(a_tise), ', l=', num2str(l_tise)]);
disp(['Number of eigenvalues k_n: ', num2str(length(k_n_10))]);
disp('First 10 k_n values (sorted by real part):');
[~, sort_idx] = sort(real(k_n_10));
k_n_sorted = k_n_10(sort_idx);
for i = 1:min(10, length(k_n_sorted))
    disp(['  k_', num2str(i), ' = ', num2str(real(k_n_sorted(i)), '%.10e'), ...
          ' + ', num2str(imag(k_n_sorted(i)), '%.10e'), 'i']);
end

% Bound states (purely imaginary k, Im(k) > 0)
bound_mask = (abs(real(k_n_10)) < 1e-10) & (imag(k_n_10) > 0);
k_bound = k_n_10(bound_mask);
disp(['Number of bound states: ', num2str(sum(bound_mask))]);
if ~isempty(k_bound)
    disp('Bound state k values:');
    for i = 1:length(k_bound)
        E_bound = -abs(k_bound(i))^2 / 2;
        disp(['  k_bound_', num2str(i), ' = ', num2str(imag(k_bound(i)), '%.10e'), 'i (E = ', num2str(E_bound, '%.10e'), ')']);
    end
end

% Test case 5.2: Same potential, l=1
l_tise = 1;
[k_n_l1, ~, ~, ~] = TISE_by_SPS(f_test, N_tise, a_tise, l_tise);
disp(' ');
disp(['Test 5.2: TISE_by_SPS with N=', num2str(N_tise), ', a=', num2str(a_tise), ', l=', num2str(l_tise)]);
[~, sort_idx] = sort(real(k_n_l1));
k_n_l1_sorted = k_n_l1(sort_idx);
disp('First 10 k_n values:');
for i = 1:min(10, length(k_n_l1_sorted))
    disp(['  k_', num2str(i), ' = ', num2str(real(k_n_l1_sorted(i)), '%.10e'), ...
          ' + ', num2str(imag(k_n_l1_sorted(i)), '%.10e'), 'i']);
end

% Test case 5.3: Larger N for convergence check
N_tise = 30;
[k_n_30, ~, ~, r_i_30] = TISE_by_SPS(f_test, N_tise, a_tise, 0);
disp(' ');
disp(['Test 5.3: TISE_by_SPS with N=', num2str(N_tise), ', a=', num2str(a_tise), ', l=0']);
bound_mask_30 = (abs(real(k_n_30)) < 1e-10) & (imag(k_n_30) > 0);
k_bound_30 = k_n_30(bound_mask_30);
disp(['Number of bound states: ', num2str(sum(bound_mask_30))]);
if ~isempty(k_bound_30)
    disp('Bound state energies:');
    for i = 1:length(k_bound_30)
        E_bound = -abs(k_bound_30(i))^2 / 2;
        disp(['  E_', num2str(i), ' = ', num2str(E_bound, '%.10e')]);
    end
end

disp('Quadrature points r_i (first 5):');
disp(r_i_30(1:5).');

%% ============================================================
%% TEST 6: calc_cross_section_by_SPS
%% ============================================================
disp(' ');
disp('=== TEST 6: Cross Section Calculation ===');
disp(' ');

% Use smaller parameters for faster testing
N_cs = 20;
a_cs = 10;
l_max_cs = 3;
E_vec_test = linspace(1e-4, 5, 50);  % Smaller grid
dtau_cs = inf;

% Same Pöschl-Teller lambda=4 potential as before
f_x_test = @(x) (-LAMBDA_PT*(LAMBDA_PT+1)/2./cosh(x).^2);

disp(['Test 6.1: calc_cross_section_by_SPS, N=', num2str(N_cs), ', a=', num2str(a_cs), ', l_max=', num2str(l_max_cs)]);
disp(['Energy grid: ', num2str(length(E_vec_test)), ' points from ', num2str(E_vec_test(1)), ' to ', num2str(E_vec_test(end))]);

[S_l_test, sigma_l_test, tau_l_test, alpha_test, k_n_l_test] = ...
    calc_cross_section_by_SPS(f_x_test, N_cs, a_cs, l_max_cs, E_vec_test, dtau_cs);

disp(' ');
disp('--- Scattering length ---');
disp(['alpha (l=0): ', num2str(alpha_test, '%.10e')]);

disp(' ');
disp('--- k_n values for each l ---');
for l = 0:l_max_cs
    k_n_l = k_n_l_test{l+1};
    disp(['l=', num2str(l), ': ', num2str(length(k_n_l)), ' eigenvalues']);
    % Show first few real resonances (positive real part, small imag)
    resonance_mask = (real(k_n_l) > 0) & (abs(imag(k_n_l)) < 0.5);
    k_res = k_n_l(resonance_mask);
    if ~isempty(k_res)
        [~, idx] = sort(real(k_res));
        k_res = k_res(idx);
        disp(['  Resonances (Re(k)>0, |Im(k)|<0.5): ', num2str(length(k_res))]);
        for i = 1:min(3, length(k_res))
            disp(['    k = ', num2str(real(k_res(i)), '%.10e'), ' + ', ...
                  num2str(imag(k_res(i)), '%.10e'), 'i']);
        end
    end
end

disp(' ');
disp('--- S-matrix at selected energies ---');
E_samples = [1, 10, 25, 50];  % indices
for i_e = E_samples
    if i_e <= length(E_vec_test)
        disp(['E = ', num2str(E_vec_test(i_e), '%.4e')]);
        for l = 0:l_max_cs
            S_val = S_l_test(i_e, l+1);
            disp(['  S_', num2str(l), ' = ', num2str(real(S_val), '%.10e'), ' + ', ...
                  num2str(imag(S_val), '%.10e'), 'i  (|S|=', num2str(abs(S_val), '%.10e'), ')']);
        end
    end
end

disp(' ');
disp('--- Cross sections at selected energies ---');
for i_e = E_samples
    if i_e <= length(E_vec_test)
        disp(['E = ', num2str(E_vec_test(i_e), '%.4e')]);
        for l = 0:l_max_cs
            disp(['  sigma_', num2str(l), ' = ', num2str(sigma_l_test(i_e, l+1), '%.10e')]);
        end
        disp(['  sigma_total = ', num2str(sum(sigma_l_test(i_e, :)), '%.10e')]);
    end
end

disp(' ');
disp('--- Time delays at selected energies ---');
for i_e = E_samples
    if i_e <= length(E_vec_test)
        disp(['E = ', num2str(E_vec_test(i_e), '%.4e')]);
        for l = 0:l_max_cs
            disp(['  tau_', num2str(l), ' = ', num2str(tau_l_test(i_e, l+1), '%.10e')]);
        end
    end
end

%% ============================================================
%% TEST 7: Full SPS Calculation (from SPS_script.m)
%% ============================================================
disp(' ');
disp('=== TEST 7: Full SPS Calculation ===');
disp(' ');

% Parameters from SPS_script.m but with reduced grid for speed
% Using Pöschl-Teller lambda=4: V = -10/cosh^2(x)
a0 = 1;
HBAR = 1;
MU = 1;
LAMBDA_PT = 4;

f_x = @(x) (-LAMBDA_PT*(LAMBDA_PT+1)/2./cosh(x).^2) * (MU*a0^2/HBAR^2);
f_1 = @(x) -1e-6*(1/4)*1./cosh(x) * (MU*a0^2/HBAR^2);
f_2 = @(x) +1e-6*(3/4)*1./cosh(x) * (MU*a0^2/HBAR^2);
f_x_1 = @(x) f_x(x) + f_1(x);
f_x_2 = @(x) f_x(x) + f_2(x);

a = 10;
N = 50;  % Reduced from 100
l_max = 5;  % Reduced from 20

% Smaller energy grid for testing
E_vec_orig = linspace(1e-6, 10, 1000);
dtau = inf;
E_vec_natural = E_vec_orig/(HBAR^2/(MU*a0^2));

disp(['Parameters: N=', num2str(N), ', a=', num2str(a), ', l_max=', num2str(l_max)]);
disp(['Energy grid: ', num2str(length(E_vec_natural)), ' points']);
disp(' ');

% Calculate for both potentials
disp('Computing potential 1...');
[S_l_1, sigma_l_1, tau_l_1, alpha_1, k_n_l_1] = ...
    calc_cross_section_by_SPS(f_x_1, N, a, l_max, E_vec_natural, dtau);
tau_l_1 = tau_l_1*(a0^2*MU./(HBAR^2))*HBAR;
alpha_1 = alpha_1*a0;

disp('Computing potential 2...');
[S_l_2, sigma_l_2, tau_l_2, alpha_2, k_n_l_2] = ...
    calc_cross_section_by_SPS(f_x_2, N, a, l_max, E_vec_natural, dtau);
tau_l_2 = tau_l_2*(a0^2*MU./(HBAR^2))*HBAR;
alpha_2 = alpha_2*a0;

disp(' ');
disp('--- Results ---');
disp(['Scattering length (alpha_1): ', num2str(alpha_1, '%.10e')]);
disp(['Scattering length (alpha_2): ', num2str(alpha_2, '%.10e')]);

% Mean lifetimes
mean_tau_1 = sum(((sigma_l_1./sum(sigma_l_1.').').*tau_l_1).');
mean_tau_2 = sum(((sigma_l_2./sum(sigma_l_2.').').*tau_l_2).');
disp(['Mean tau_1 at E=E_vec(1): ', num2str(mean_tau_1(1), '%.10e')]);
disp(['Mean tau_2 at E=E_vec(1): ', num2str(mean_tau_2(1), '%.10e')]);

% Rate coefficient calculation
kbT_natural = 5;
f_E = (2/sqrt(pi))*(1/kbT_natural)^(3/2)*exp(-E_vec_natural/kbT_natural);
sigma_12 = (a0^2*pi./(2*E_vec_natural)).*sum(((2*(0:l_max)+1).*abs((S_l_1-S_l_2)/2).^2).');
Gamma_12 = 1e6 * trapz(E_vec_natural, sqrt(2*E_vec_natural).*sigma_12.*f_E);

disp(['Rate coefficient Gamma_12: ', num2str(Gamma_12, '%.10e'), ' cm^3/sec']);

disp(' ');
disp('--- S-matrix values at specific energies ---');
E_check_idx = [1, 100, 500, 1000];
for i_e = E_check_idx
    if i_e <= length(E_vec_natural)
        disp(['E = ', num2str(E_vec_natural(i_e), '%.6e')]);
        disp(['  S_0(pot1) = ', num2str(real(S_l_1(i_e,1)), '%.10e'), ' + ', ...
              num2str(imag(S_l_1(i_e,1)), '%.10e'), 'i']);
        disp(['  S_0(pot2) = ', num2str(real(S_l_2(i_e,1)), '%.10e'), ' + ', ...
              num2str(imag(S_l_2(i_e,1)), '%.10e'), 'i']);
    end
end

disp(' ');
disp('--- Cross sections at specific energies ---');
for i_e = E_check_idx
    if i_e <= length(E_vec_natural)
        disp(['E = ', num2str(E_vec_natural(i_e), '%.6e')]);
        disp(['  sigma_total(pot1) = ', num2str(sum(sigma_l_1(i_e,:)), '%.10e')]);
        disp(['  sigma_total(pot2) = ', num2str(sum(sigma_l_2(i_e,:)), '%.10e')]);
    end
end

%% ============================================================
%% TEST 8: Numerical consistency checks
%% ============================================================
disp(' ');
disp('=== TEST 8: Numerical Consistency ===');
disp(' ');

% Check |S|^2 = 1 (unitarity)
S_magnitude_l0 = abs(S_l_1(:,1)).^2;
disp('Test 8.1: S-matrix unitarity |S_l|^2 = 1');
disp(['  max(|S_0|^2 - 1) for pot1 = ', num2str(max(abs(S_magnitude_l0 - 1)), '%.10e')]);
disp(['  mean(|S_0|^2) for pot1 = ', num2str(mean(S_magnitude_l0), '%.10e')]);

% Check that bound state energies are consistent
disp(' ');
disp('Test 8.2: Bound state energies from k_n_l');
for l = 0:min(2, l_max)
    k_n = k_n_l_1{l+1};
    % Bound states: purely imaginary with positive imaginary part
    bound_mask = (abs(real(k_n)) < 1e-8) & (imag(k_n) > 0);
    k_bound = k_n(bound_mask);
    if ~isempty(k_bound)
        [~, idx] = sort(-imag(k_bound));  % Sort by binding energy
        k_bound = k_bound(idx);
        disp(['  l=', num2str(l), ' bound states:']);
        for i = 1:length(k_bound)
            E = -abs(k_bound(i))^2 / 2;
            disp(['    E_', num2str(i), ' = ', num2str(E, '%.10e')]);
        end
    else
        disp(['  l=', num2str(l), ': no bound states found']);
    end
end

%% ============================================================
%% Summary JSON-like output for easy parsing
%% ============================================================
disp(' ');
disp('=== SUMMARY (JSON-like format) ===');
disp('{');
disp('  "test_jacobi_polynomial": {');
disp(['    "alpha": ', num2str(alpha), ',']);
disp(['    "beta": ', num2str(beta), ',']);
disp('    "x": [-1, -0.5, 0, 0.5, 1],');
disp('    "P_0": [1, 1, 1, 1, 1],');
JP_row = JP(:,2)';
disp(['    "P_1": [', num2str(JP_row(1), '%.10e'), ', ', num2str(JP_row(2), '%.10e'), ', ', ...
      num2str(JP_row(3), '%.10e'), ', ', num2str(JP_row(4), '%.10e'), ', ', num2str(JP_row(5), '%.10e'), ']']);
disp('  },');

disp('  "test_quadrature_N5_l0": {');
disp(['    "nodes": [', strjoin(arrayfun(@(x) num2str(x, '%.15e'), x_i, 'UniformOutput', false), ', '), '],']);
disp(['    "weights": [', strjoin(arrayfun(@(x) num2str(x, '%.15e'), omega_i, 'UniformOutput', false), ', '), ']']);
disp('  },');

disp('  "test_scattering_length": {');
disp(['    "alpha_1": ', num2str(alpha_1, '%.15e'), ',']);
disp(['    "alpha_2": ', num2str(alpha_2, '%.15e')]);
disp('  },');

disp('  "test_rate_coefficient": {');
disp(['    "kbT_natural": ', num2str(kbT_natural), ',']);
disp(['    "Gamma_12_cm3_per_sec": ', num2str(Gamma_12, '%.15e')]);
disp('  }');
disp('}');

disp(' ');
disp('================================================================');
disp('TEST SUITE COMPLETE');
disp('================================================================');
