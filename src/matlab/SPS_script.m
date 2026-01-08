%% SPS code demonstration
% Everything is in natural units unless stated otherwise
a0 = 1;
HBAR = 1;
MU = 1;
% If you have the potential in [J], set a0, HBAR, MU to the corresponding
% units and let things run out as normal, conversions should be taken care
% of along the way

% Definition of two single channel potentials
f_x = @(x) (-21./cosh(x).^2) * (MU*a0^2/HBAR^2);
f_1 = @(x) -1e-6*(1/4)*1./cosh(x) * (MU*a0^2/HBAR^2);
f_2 = @(x) +1e-6*(3/4)*1./cosh(x) * (MU*a0^2/HBAR^2);
f_x_1 = @(x) f_x(x) + f_1(x);
f_x_2 = @(x) f_x(x) + f_2(x);

% SPS calculation parameters (For k_n and psi)
a = 10; % Cutoff radius
N = 100; % Number of basis functions
l_max = 20; % Maximum value of l to calculate

% Parameters for post-SPS calculations (Take k_n and construct S_l(E), tau(E), etc.)
E_vec_orig = linspace(1e-6,30,6e5); % Original energy grid to calculate on [J], the final energy grid is different to this, explained below
dtau = inf; % 1e-9; % Lifetime limiting parameter in [sec]
E_vec_orig_natural_units = E_vec_orig/(HBAR^2/(MU*a0^2));

[~, ~, ~, ~, ~, E_vec_1] = calc_cross_section_by_SPS(f_x_1, N, a, l_max, E_vec_orig_natural_units, dtau);
[~, ~, ~, ~, ~, E_vec_2] = calc_cross_section_by_SPS(f_x_2, N, a, l_max, E_vec_orig_natural_units, dtau);
E_vec_natural_units = unique([E_vec_1, E_vec_2]);
% The new E_vec_natural_units is a non-uniformly sampled grid that captures
% all resonances of the potential, and also has 2000 equally spaced points
% in [E_vec_orig(1), E_vec_orig(end)]. Ugly and hardcoded, I know.

%% Actual calculation
[S_l_1, sigma_l_1, tau_l_1, alpha_1, k_n_l_1] = calc_cross_section_by_SPS(f_x_1, N, a, l_max, E_vec_natural_units, dtau);
[S_l_2, sigma_l_2, tau_l_2, alpha_2, k_n_l_2] = calc_cross_section_by_SPS(f_x_2, N, a, l_max, E_vec_natural_units, dtau);
tau_l_1 = tau_l_1*(a0^2*MU./(HBAR^2))*HBAR;
alpha_1 = alpha_1*a0;
tau_l_2 = tau_l_2*(a0^2*MU./(HBAR^2))*HBAR;
alpha_2 = alpha_2*a0;
mean_tau_4s_S = sum(((sigma_l_1./sum(sigma_l_1.').').*tau_l_1).');
mean_tau_4s_S_binary = sum(((sigma_l_1./sum(sigma_l_1.').').*medfilt1(tau_l_1, 3000)).');
mean_tau_4s_T = sum(((sigma_l_2./sum(sigma_l_2.').').*tau_l_2).');  
mean_tau_4s_T_binary = sum(((sigma_l_2./sum(sigma_l_2.').').*medfilt1(tau_l_2, 3000)).');

kb = 1.38e-23;
T = 273;
% kbT_natural = (kb*T) / (HBAR^2/(MU*a0^2)); % kbT in natural units
kbT_natural = 5
f_E = (2/sqrt(pi))*(1/kbT_natural)^(3/2)*exp(-E_vec_natural_units/kbT_natural);
sigma_12 = (a0^2*pi./(2*E_vec_natural_units)).*sum(((2*(0:l_max)+1).*abs((S_l_1-S_l_2)/2).^2).');
Gamma_12 = 1e6*trapz((HBAR^2/(MU*a0^2))*E_vec_natural_units, ...
  (HBAR^2/(MU*a0^2))*sqrt(2/MU)*E_vec_natural_units.*sigma_12.*f_E); % In cm^3/sec
% Rate coefficient: Gamma = <sigma * v> where v = sqrt(2E/MU)
% In natural units with MU=1: v = sqrt(2E)
Gamma_12 = 1e6 * trapz(E_vec_natural_units, sqrt(2*E_vec_natural_units).*sigma_12.*f_E); % In cm^3/sec

%% Display Results
disp(' ');
disp('========== SPS Calculation Results ==========');
disp(' ');
disp(['kT: ', num2str(kb * T)]);
disp(['Cutoff radius (a): ', num2str(a)]);
disp(['Number of basis functions (N): ', num2str(N)]);
disp(['Maximum l value: ', num2str(l_max)]);
disp(['Energy grid points: ', num2str(length(E_vec_natural_units))]);
disp(' ');
disp('--- Potential 1 (f_x_1) ---');
disp(['  Scattering length (alpha_1): ', num2str(alpha_1(1), '%.6e')]);
disp(['  Mean lifetime (mean_tau_4s_S): ', num2str(mean_tau_4s_S(1), '%.6e')]);
disp(['  Mean lifetime binary (mean_tau_4s_S_binary): ', num2str(mean_tau_4s_S_binary(1), '%.6e')]);
disp(' ');
disp('--- Potential 2 (f_x_2) ---');
disp(['  Scattering length (alpha_2): ', num2str(alpha_2(1), '%.6e')]);
disp(['  Mean lifetime (mean_tau_4s_T): ', num2str(mean_tau_4s_T(1), '%.6e')]);
disp(['  Mean lifetime binary (mean_tau_4s_T_binary): ', num2str(mean_tau_4s_T_binary(1), '%.6e')]);
disp(' ');
disp('--- Rate Coefficient ---');
disp(['  Gamma_12: ', num2str(Gamma_12, '%.6e'), ' cm^3/sec']);
disp(' ');
disp('=============================================');

%% Optional: Save results to file
save('SPS_results.mat', 'S_l_1', 'S_l_2', 'sigma_l_1', 'sigma_l_2', 'tau_l_1', 'tau_l_2', ...
     'alpha_1', 'alpha_2', 'k_n_l_1', 'k_n_l_2', 'E_vec_natural_units', 'Gamma_12', ...
     'mean_tau_4s_S', 'mean_tau_4s_T', 'mean_tau_4s_S_binary', 'mean_tau_4s_T_binary');
disp('Results saved to SPS_results.mat');
