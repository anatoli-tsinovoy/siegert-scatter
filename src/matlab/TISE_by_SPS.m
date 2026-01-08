function [k_n, c_ctilde_zeta, eigenmodes_in_ri_basis, r_i, CVC, Cf, z_l_p, C, V2_tilde, f] = ...
    TISE_by_SPS(V, N, a, l, V_2)
%% Defining the analytic potential function
% MU = 4.6243e-27;
% HBAR = 1.0546e-34;
% V_eff = @(z) V(z) + (HBAR^2/MU)*l*(l+1)./(2*z.^2);
V_eff = @(z) V(z) + l*(l+1)./(2*z.^2);

%% Defining the basis functions
jacobiP_normalization = @(n) 1./sqrt((2.^(1 + 2*l))./(1 + 2*n + 2*l));

[omega_i, x_i] = get_gaussian_quadrature(N,0,2*l,-1,1);
JP_xi_n = j_polynomial(N+1, N-1, 0, 2*l, [x_i;1]);
phi_n_xi = (jacobiP_normalization((0:(N-1))).*((1+x_i).^l).*JP_xi_n(1:N,:)).';
phi_n_1 = (jacobiP_normalization((0:(N-1))).*((1+1).^l).*JP_xi_n(N+1,:)).';

T_n_i = bsxfun(@times, phi_n_xi, (sqrt(omega_i)./(1+x_i).^l).');

%% Constructing the matrices H_tilde_l and F
calc_type = 'Gauss-Jacobi';
switch calc_type
    case 'Gauss-Jacobi'
        pi_i_1 = T_n_i.'*phi_n_1;
        f_i_a = sqrt(2/a)*((1+1)./(1+x_i)).*(pi_i_1);
        F = f_i_a*f_i_a.';
        U_tilde_l = diag(V_eff(a*(1+x_i)/2));
        
        K_tilde_nm_phi_frac = zeros(N,1);
        diag_K_tilde_nm_phi = zeros(N,1);
        for n=1:N
            K_tilde_nm_phi_frac(n) = ...
                (2*sum(phi_n_1(1:(n-1)).^2)+phi_n_1(n)^2-(1/2))*phi_n_1(n);
            diag_K_tilde_nm_phi(n) = 2*phi_n_1(n)^2*(sum(phi_n_1(1:(n-1)).^2)) + ...
                (1/2)*(phi_n_1(n)^2-(1/2))^2;
        end
        K_tilde_nm_phi = K_tilde_nm_phi_frac*phi_n_1.';
        K_tilde_nm_phi = triu(K_tilde_nm_phi) + triu(K_tilde_nm_phi).';
        K_tilde_nm_phi = K_tilde_nm_phi - diag(diag(K_tilde_nm_phi)) + ...
            diag(diag_K_tilde_nm_phi);
        
        % I checked, K_tilde_nm_phi is calculated properly despite all of the manipulations
        % for n=1:N
        %     for m=1:N
        %         switch sign(n-m)
        %             case -1 % n<m
        %                 exact_K_tilde_nm_phi(n,m) = phi_n_1(n)*phi_n_1(m)* ...
        %                     (2*sum(phi_n_1(1:(n-1)).^2)+phi_n_1(n)^2-(1/2));
        %             case 0  % n=m
        %                 exact_K_tilde_nm_phi(n,m) = 2*phi_n_1(n)^2* ...
        %                     sum(phi_n_1(1:(n-1)).^2) + (1/2)* ...
        %                     (phi_n_1(n)^2-(1/2))^2;
        %             case 1  % n>m
        %                 exact_K_tilde_nm_phi(n,m) = phi_n_1(m)*phi_n_1(n)* ...
        %                     (2*sum(phi_n_1(1:(m-1)).^2)+phi_n_1(m)^2-(1/2));
        %         end
        %     end
        % end
        
        % I checked, K_tilde_l is calculated properly when compared to
        % numerical calculation of the derivative term
        % x_vec = linspace(-1,1, 1000);
        % x_vec(1) = [];
        % r_vec = (a/2)*(x_vec+1);
        % dx = (max(x_vec)-min(x_vec))/length(x_vec);
        % dr = (a/2)*dx;
        % JP_x_vec = j_polynomial(length(x_vec), N-1, 0, 2*l, x_vec.');
        % phi_n_x_vec = (jacobiP_normalization((0:(N-1))).*((1+x_vec.').^l).*JP_x_vec).';
        % pi_i_x_vec = T_n_i.'*phi_n_x_vec;
        % f_i_x_vec = sqrt(2/a)*((1+x_vec)./(1+x_i)).*(pi_i_x_vec);
        % dfi_dr = (gradient(f_i_x_vec)/dr);
        % dfi_dr_xi = (interp1(x_vec, dfi_dr.', x_i).');
        % K_tilde_l_alt = zeros(N,N);
        % for i=1:N
        %     for j=1:N
        %         K_tilde_l_alt(i,j) = ...
        %             (a/4)*sum(dfi_dr_xi(i,:).*dfi_dr_xi(j,:).*(omega_i./(1+x_i).^(2*l)).');
        %     end
        % end
        % K_tilde_l_no_GQ = zeros(N,N);
        % for i=1:N
        %     for j=1:N
        %         K_tilde_l_no_GQ(i,j) = (1/2)*trapz(r_vec, dfi_dr(i,:).*dfi_dr(j,:));
        %     end
        % end
        
        K_tilde_l = (((2/a)*1./(1+x_i))*((2/a)*1./(1+x_i)).').* ...
            (T_n_i.'*K_tilde_nm_phi*T_n_i + (1)*(pi_i_1*pi_i_1.'));
        
        %     case 'trapz'
        %         x_vec = linspace(-1,1, 1000);
        %         x_vec(1) = [];
        %         r_vec = (a/2)*(x_vec+1);
        %         phi_n_x_vec = zeros(N,length(x_vec));
        %         for n=1:N
        %             phi_n_x_vec(n,:) = phi(n,x_vec);
        %         end
        %         pi_i_x_vec = T_n_i.'*phi_n_x_vec;
        %         % f_i_x_vec = sqrt(2/a)*((1+x_vec)./(1+x_i)).*(pi_i_x_vec);
        %         f_i_x_vec = sqrt(2/a)*((1+x_vec)./(1+x_i)).*(pi_i_x_vec);
        %         % f_i_x_vec = f_i_x_vec./sqrt(diag(f_i_x_vec*f_i_x_vec.'));
        %         dx = (max(x_vec)-min(x_vec))/length(x_vec);
        %         dr = (a/2)*dx;
        %         dfi_dr = gradient(f_i_x_vec,2)/dx * (dx/dr);
        %
        %         f_i_a = f_i_x_vec(:,end);
        %         F = f_i_a*f_i_a.';
        %
        %         K_tilde_l = zeros(N,N);
        %         U_tilde_l = zeros(N,N);
        %         for i=1:N
        %             for j=1:N
        %                 K_tilde_l(i,j) = (1/2)*trapz(r_vec,dfi_dr(i,:).*dfi_dr(j,:));
        %                 U_tilde_l(i,j) = trapz(r_vec,f_i_x_vec(i,:).*V_eff(r_vec).*f_i_x_vec(j,:));
        %             end
        %         end
end

H_tilde_l = U_tilde_l + K_tilde_l;

%% Constructing the entire matrix to be diagonalized
z_l_p = calc_z_l(l,false).';
if mod(l,2) == 0
    pure_real_z = [];
else
    pure_real_z = z_l_p(imag(z_l_p)==0);
end
z_l_p = [pure_real_z;
    sort(z_l_p(imag(z_l_p)~=0), 'ComparisonMethod', 'real')];

if mod(l,2) == 0
    U = [];
else
    U = 1;
end
for i=1:floor(l/2)
    U = blkdiag(U, [1,1i;1,-1i]);
end
bigU = blkdiag(eye(N), eye(N), U);

z_f_i_a_mat = -(z_l_p.*f_i_a.')/a;
diag_z_mat = -diag(z_l_p)/a;
diag_rz_mat = diag(-1./z_l_p);

tot_mat = [zeros(N,N), diag(ones(N,1)), zeros(N,l); ...
    -2*H_tilde_l, F, repmat(f_i_a, 1, l)/a; ...
    z_f_i_a_mat, zeros(l, N), diag_z_mat];

W = [-F, diag(ones(N,1)), zeros(N,l);
    diag(ones(N,1)), zeros(N,N), zeros(N,l);
    zeros(l,N), zeros(l,N), diag_rz_mat];

mat_34 = [-2*H_tilde_l, zeros(N,N), repmat(f_i_a/a, 1, l);
    zeros(N,N), diag(ones(N,1)), zeros(N,l);
    repmat(f_i_a.'/a, l, 1), zeros(l,N), diag(ones(1,l))/a];

real_only = true;
if real_only
    tot_mat = real(inv(bigU)*tot_mat*bigU);
    W = real(bigU.'*W*bigU);
    mat_34 = real(inv(bigU)*mat_34*bigU);
end


%% Diagonalizing the resulting matrix
[c_ctilde_zeta, lambda] = eig(tot_mat, 'vector');
% [c_ctilde_zeta, lambda] = eig(mat_34, W, 'vector');
k_n = (-1i*lambda);

if nargout > 1 || nargin > 4
    c_ctilde_zeta_norm_under_W = ...
        diag(c_ctilde_zeta.'*W*c_ctilde_zeta);
    c_ctilde_zeta_normalized = ...
        c_ctilde_zeta.*sqrt(2*lambda./c_ctilde_zeta_norm_under_W).';
    % Normalization condition is c_nT*W*c_m = 2*lambda * delta_nm
    pi_i_xi = T_n_i.'*phi_n_xi;
    f_i_xi = sqrt(2/a)*((1+x_i).'./(1+x_i)).*(pi_i_xi);
    % eigenmodes_in_ri_basis = eigenmodes_in_jacobi_basis(1:N,:).'*f_i_xi;
    eigenmodes_in_ri_basis = c_ctilde_zeta_normalized(1:N,:).'*f_i_xi;
    r_i = (a/2)*(x_i+1);
end

if nargin > 4
    V2_tilde = diag(V_2(a*(1+x_i)/2));
    C = c_ctilde_zeta_normalized(1:N,:).';
    CVC = C*V2_tilde*C.';
    Cf = C*f_i_a;
    f = f_i_a;
end
end