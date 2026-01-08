%% SPS Quick Validation Test
% Minimal test that runs fast (<5 sec) for quick sanity checks
% Outputs key numerical values that must match Python implementation

format long e;
disp('=== SPS QUICK VALIDATION TEST ===');
disp(' ');

%% Test 1: Jacobi polynomial spot check
x = [0; 0.5; 1];
JP = j_polynomial(3, 2, 0, 4, x);
disp('Jacobi P_2(0,4) at x=0,0.5,1:');
disp(JP(:,3).');

%% Test 2: Gauss-Jacobi quadrature
[w, nodes] = get_gaussian_quadrature(5, 0, 0, -1, 1);
disp('Gauss-Legendre N=5 nodes:');
disp(nodes.');

%% Test 3: Bessel zeros
z3 = calc_z_l(3, false);
disp('Spherical Bessel zeros l=3:');
for i = 1:length(z3)
    disp(['  ', num2str(real(z3(i)), '%.12e'), ' + ', num2str(imag(z3(i)), '%.12e'), 'i']);
end

%% Test 4: TISE eigenvalues for Pöschl-Teller potential
% V(x) = -lambda*(lambda+1)/2 / cosh^2(x), lambda=4 -> V = -10/cosh^2(x)
% Full-line bound states: E_n = -(lambda-n)^2/2 for n=0,1,2,3
%   E_0=-8 (even), E_1=-4.5 (odd), E_2=-2 (even), E_3=-0.5 (odd)
%
% For RADIAL l=0 with psi(0)=0: only ODD parity states survive
LAMBDA_PT = 4;
a_pt = 20;
f_pt_radial = @(r) -LAMBDA_PT*(LAMBDA_PT+1)/2 ./ cosh(r).^2;  % at origin
[k_n, ~, ~, ~] = TISE_by_SPS(f_pt_radial, 100, a_pt, 0);

bound = k_n(abs(real(k_n)) < 1e-6 & imag(k_n) > 0.5);
E_bound = -abs(bound).^2 / 2;
E_bound = sort(E_bound, 'descend');
disp('Test 4a: Radial l=0 bound states (psi(0)=0, odd parity only):');
disp(E_bound.');
disp('Exact: -0.5, -4.5');

% Test 4b: Full potential (shifted to capture all states)
f_pt_full = @(r) -LAMBDA_PT*(LAMBDA_PT+1)/2 ./ cosh(r - a_pt/2).^2;
[k_n_full, ~, ~, ~] = TISE_by_SPS(f_pt_full, 100, a_pt, 0);

bound_full = k_n_full(abs(real(k_n_full)) < 1e-6 & imag(k_n_full) > 0.5);
E_bound_full = -abs(bound_full).^2 / 2;
E_bound_full = sort(E_bound_full, 'descend');
disp('Test 4b: Full Pöschl-Teller (centered at a/2):');
disp(E_bound_full.');
disp('Exact: -0.5, -2, -4.5, -8');

%% Test 5: Scattering length (radial potential at origin)
[~, ~, ~, alpha, ~] = calc_cross_section_by_SPS(f_pt_radial, 100, a_pt, 0, [0.01], inf);
disp(['Scattering length: ', num2str(alpha, '%.10e')]);

%% Test 6: S-matrix at E=1 (radial potential)
E_test = 1;
[S, sigma, tau, ~, ~] = calc_cross_section_by_SPS(f_pt_radial, 100, a_pt, 2, E_test, inf);
disp(' ');
disp('At E=1:');
disp(['  S_0 = ', num2str(real(S(1)), '%.10e'), ' + ', num2str(imag(S(1)), '%.10e'), 'i']);
disp(['  sigma_0 = ', num2str(sigma(1), '%.10e')]);
disp(['  |S_0| = ', num2str(abs(S(1)), '%.15e')]);

disp(' ');
disp('=== VALIDATION COMPLETE ===');
