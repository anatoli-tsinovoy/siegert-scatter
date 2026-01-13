function [S_J, sigma_J, tau_J, alpha, k_n_J] = calc_cross_section_by_SPS_with_total_AM(f_x, N, a, J_min, J_max, E_vec, Lambda, Sigma, A_SO, S)
%% Input:
% f_x      function_handle Radial potential in a.u.
% N        [1,1] Number of basis functions to employ
% a        [1,1] Cutoff for the potential (a.u.)
% J_max    [1,1] Maximal value of l to calculate for
%                (J_max<=60+1/2 due to hard-coded table of zeros)
% E_vec    [1,M] Energies to calculate the scattering at in a.u.

%% Output:
% Gamma_SE  [1,1] Exchange rate per unit density (cm^3/sec)

k_n_J = cell(1,1+round(J_max-1/2));
for Jtag=round(J_min-1/2):1:round(J_max-1/2)
    Jtag
    f_x_Jtag = build_f_x_Jtag_by_total_AM(f_x, Lambda, Sigma, Jtag, A_SO, S);
    [k_n, ~] = TISE_by_SPS(f_x_Jtag, N, a, Jtag);
    k_n_J{Jtag+1} = k_n;
end

alpha = real(a+sum(1i./k_n_J{1})); % Zero scattering length

k_vec = sqrt(2*E_vec); % k vector in natural units
% S_l_k = @(k_n,k) exp(-2i*k*a).*prod((k_n+k)./(k_n-k));
% delta_l_k = @(k_n,k) -a + sum(-((imag(k_n).*(k.^2 + abs(k_n).^2))./ ...
%     (k.^4 + abs(k_n).^4 + 2*k.^2.*(abs(k_n).^2 - 2*real(k_n).^2))))
sigma_J = zeros(length(k_vec), 1+round(J_max-1/2));
tau_J = zeros(length(k_vec), 1+round(J_max-1/2));
S_J = zeros(length(k_vec), 1+round(J_max-1/2));
for Jtag=round(J_min-1/2):1:round(J_max-1/2)
    S_l_k = exp(-2i*k_vec*a);
    d_delta_l_k_dk = -a;
    for k_ind=1:length(k_n_J{Jtag+1})
        S_l_k = S_l_k.*(k_n_J{Jtag+1}(k_ind)+k_vec)./(k_n_J{Jtag+1}(k_ind)-k_vec);
        d_delta_l_k_dk = d_delta_l_k_dk + ...
            (-(imag(k_n_J{Jtag+1}(k_ind)).*((imag(k_n_J{Jtag+1}(k_ind))).^2 + k_vec.^2 + ...
            (real(k_n_J{Jtag+1}(k_ind))).^2))./ ...
            (k_vec.^4 + 2*k_vec.^2.*(imag(k_n_J{Jtag+1}(k_ind)) - real(k_n_J{Jtag+1}(k_ind))).* ...
            (imag(k_n_J{Jtag+1}(k_ind)) + real(k_n_J{Jtag+1}(k_ind))) + ((imag(k_n_J{Jtag+1}(k_ind))).^2 + ...
            (real(k_n_J{Jtag+1}(k_ind))).^2).^2));
    end
    
    sigma_J(:,Jtag+1) = (2*Jtag+1)*(pi./k_vec.^2).*abs(1-S_l_k).^2;
    tau_J(:,Jtag+1) = d_delta_l_k_dk./k_vec;
    S_J(:,Jtag+1) = S_l_k;
end
end

function f_x_Jtag = build_f_x_Jtag_by_total_AM(f_x, Lambda, Sigma, Jtag, A_SO, S)
f_x_Jtag = @(z) f_x(z) + ...
    (1./(2*z.^2)).* ...
    ((3/4) + Jtag + ... % Correction for half-integer AM
    S*(S+1) + 2*Lambda*Sigma + Lambda^2 - 2*(Lambda+Sigma)^2) + ... % Rotational Hamiltonian contribution for Hund's case (a)
    A_SO*Lambda*Sigma; % SO contribution
end