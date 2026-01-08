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
E_vec_orig = linspace(0,30,6e5); % Original energy grid to calculate on [J], the final energy grid is different to this, explained below
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
f_E = (2/sqrt(pi))*(1/(kb*T)^(3/2))*exp(-E_vec_natural_units*(HBAR^2/(MU*a0^2))/(kb*T));
sigma_12 = (a0^2*pi./(2*E_vec_natural_units)).*sum(((2*(0:l_max)+1).*abs((S_l_1-S_l_2)/2).^2).');
Gamma_12 = 1e6*trapz((HBAR^2/(MU*a0^2))*E_vec_natural_units, ...
    (HBAR^2/(MU*a0^2))*sqrt(2/MU)*E_vec_natural_units.*sigma_12.*f_E); % In cm^3/sec