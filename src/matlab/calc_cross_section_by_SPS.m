function [S_l, sigma_l, tau_l, alpha, k_n_l, E_vec_out] = calc_cross_section_by_SPS(f_x, N, a, l_max, E_vec, dtau, V_2)
%% Input:
% f_x      function_handle Radial potential in a.u.
% N        [1,1] Number of basis functions to employ
% a        [1,1] Cutoff for the potential (a.u.)
% l_max    [1,1] Maximal value of l to calculate for
%                (l_max<=60 due to hard-coded table of zeros)
% E_vec    [1,M] Energies to calculate the scattering at in a.u.

%% Output:
% Gamma_SE  [1,1] Exchange rate per unit density (cm^3/sec)

if nargin > 6
    k_n_l = cell(1,1+l_max);
    CVC_l = cell(1,1+l_max);
    Cf_l = cell(1,1+l_max);
    C_l = cell(1, 1+l_max);
    V_l = cell(1, 1+l_max);
    f_l = cell(1, 1+l_max);
end
z_l_p_l = cell(1,1+l_max);
for l=0:1:l_max
    l
    if nargin > 6
        [k_n, ~, ~, ~, CVC, Cf, z_l_p, C, V, f] = TISE_by_SPS(f_x, N, a, l, V_2);
        CVC_l{l+1} = CVC;
        Cf_l{l+1} = Cf;
        z_l_p_l{l+1} = z_l_p;
        C_l{l+1} = C;
        V_l{l+1} = V;
        f_l{l+1} = f;
    else
        [k_n, ~, ~, ~] = TISE_by_SPS(f_x, N, a, l);
    end
    k_n_l{l+1} = k_n;
end

k_vec_ext = sqrt(2*E_vec);
dGamma = 1.2279e-13/dtau;
if nargout < 6
    k_vec_out = k_vec_ext; % k vector in natural units
else
    k_vec_out = [];
    for l=0:l_max
        for i=1:length(k_n_l{l+1})
            if real(k_n_l{l+1}(i)) > 0 && abs(imag(k_n_l{l+1}(i))) < 0.0422
                width = max(dGamma/real(k_n_l{l+1}(i)), abs(imag(k_n_l{l+1}(i))));
                min_val = real(k_n_l{l+1}(i))-3*width;
                far_min_val = real(k_n_l{l+1}(i))-10*width;
                super_far_min_val = real(k_n_l{l+1}(i))-100*width;
                ultra_far_min_val = real(k_n_l{l+1}(i))-100*width;
                max_val = real(k_n_l{l+1}(i))+3*width;
                far_max_val = real(k_n_l{l+1}(i))+10*width;
                super_far_max_val = real(k_n_l{l+1}(i))+100*width;
                ultra_far_max_val = real(k_n_l{l+1}(i))+1000*width;
                k_vec_out = [k_vec_out;
                    linspace(ultra_far_min_val, super_far_min_val, 19).';
                    linspace(super_far_min_val, far_min_val, 19).';
                    linspace(far_min_val, min_val, 19).';
                    linspace(min_val, max_val, 19).';
                    linspace(max_val, far_max_val, 19).';
                    linspace(far_max_val, super_far_max_val, 19).';
                    linspace(super_far_max_val, ultra_far_max_val, 19).'];
            end
        end
    end
    k_vec_out = [k_vec_out; linspace(k_vec_ext(1), k_vec_ext(end), 2000).'];
    k_vec_out = sort(unique(k_vec_out(k_vec_out >= k_vec_ext(1) & k_vec_out <= k_vec_ext(end))));
    k_vec_out = k_vec_out.';
    E_vec_out = k_vec_out.^2/2;
end

if nargin > 6
    B_vals = [-flip(logspace(-5, -1, 50)), 0, logspace(-5, -1, 50)];
    k_vecs = cell(1,length(B_vals));
    sigma_aug_B = cell(1,length(B_vals));
    for B_ind = 1:length(B_vals)
        B_ind
        temp_k_vec = [k_vec_out, sqrt(2*(k_vec_out.^2/2-B_vals(B_ind)))];
        temp_k_vec = temp_k_vec(imag(temp_k_vec)==0);
        k_vecs{B_ind} = unique(temp_k_vec);
        T_2_l_V = zeros(length(k_vecs{B_ind}), l_max+1);
        % T_2_l_V_alt = zeros(length(k_vec_out), l_max+1);
        % T_2_l_T = zeros(length(k_vec_out), l_max+1);
        % T_2_l_T2 = zeros(length(k_vec_out), l_max+1);
        for l=0:l_max
            l
            e_l_z = @(z) (polyval(poly(z_l_p_l{l+1}), -1i*z)./(-1i*z).^l).*exp(1i*z);
            % k_n_l_aug = real(k_n_l{l+1}) + 1i*(imag(k_n_l{l+1}) - ...
            %     (real(k_n_l{l+1}) ~= 0).*dGamma./(abs(real(k_n_l{l+1}))+(real(k_n_l{l+1}) == 0)));
            k_n_l_aug = real(k_n_l{l+1}) + 1i*(imag(k_n_l{l+1}) - ...
                (real(k_n_l{l+1}) ~= 0).*dGamma./((abs(k_n_l{l+1}))+(real(k_n_l{l+1}) == 0)));
            % M = @(k) ((-1i*k/(e_l_z(k*a)))./(k_n_l{l+1}.*(k_n_l{l+1}-k)));
            M_aug = @(k) ((-1i*k/(e_l_z(k*a)))./(k_n_l_aug.*(k_n_l_aug-k)));
            M_tag = @(k) M_aug(sqrt(2*(k^2/2+B_vals(B_ind))));
            for i=1:length(k_vecs{B_ind})
                if k_vecs{B_ind}(i)^2/2+B_vals(B_ind) > 0
                    % if mod(i,1000) == 0
                    %     i
                    % end
                    % M_i = M(k_vec_out(i));
                    M_aug_i = M_aug(k_vecs{B_ind}(i));
                    M_tag_i = M_tag(k_vecs{B_ind}(i));
                    % MCf_i = M_i.*Cf_l{l+1};
                    M_augCf_i = M_aug_i.*Cf_l{l+1};
                    M_tagCf_i = M_tag_i.*Cf_l{l+1};
                    
                    T_2_l_V(i,l+1) = (-1/(k_vecs{B_ind}(i)))*((M_augCf_i.'*CVC_l{l+1}*M_tagCf_i));
                    % T_2_l_V_aug(i,l+1) = (-1/(k_vec_out(i)))*((M_augCf_i.'*CVC_l{l+1}*M_augCf_i));
                    % T_2_T_alt(i,l+1) = (-1/(k_vec_out(i)))*((MCf_i.'*pinv(eye(size(CVC_l{l+1})) - CVC_l{l+1}.*M_i)*CVC_l{l+1}*MCf_i));
                    
                    % CMC = C_l{l+1}.'*(M_i.*C_l{l+1});
                    % CMCf_i = CMC*f_l{l+1};
                    % T_2_l_T(i,l+1) = (-1/(k_vec_out(i)))*((CMCf_i.'*(pinv(eye(size(V_l{l+1}))-V_l{l+1}.*CMC))*V_l{l+1}*CMCf_i));
                    % T_2_l_T2(i,l+1) = (-1/(k_vec_out(i)))*((CMCf_i.'*(pinv(eye(size(V_l{l+1}))-(V_l{l+1}.*CMC)*(V_l{l+1}.*CMC)))*V_l{l+1}*CMCf_i));
                    % T_2_l_V_alt(i,l+1) = (-1/(k_vec_out(i)))*((CMCf_i.'*V_l{l+1}*CMCf_i));
                end
            end
        end
        sigma_aug_B{B_ind} = (pi*a0^2./k_vecs{B_ind}.^2).*sum(((2*(0:l_max)+1).*abs(T_2_l_V).^2).');
    end
    
end





alpha = real(a+sum(1i./k_n_l{1})); % Zero scattering length

% S_l_k = @(k_n,k) exp(-2i*k*a).*prod((k_n+k)./(k_n-k));
% delta_l_k = @(k_n,k) -a + sum(-((imag(k_n).*(k.^2 + abs(k_n).^2))./ ...
%     (k.^4 + abs(k_n).^4 + 2*k.^2.*(abs(k_n).^2 - 2*real(k_n).^2))))
sigma_l = zeros(length(k_vec_out), l_max+1);
tau_l = zeros(length(k_vec_out), l_max+1);
S_l = zeros(length(k_vec_out), l_max+1);
if nargout >= 5
    for l=0:l_max
        l
        k_n_l_aug = real(k_n_l{l+1}) + 1i*(imag(k_n_l{l+1}) - ...
            (real(k_n_l{l+1}) ~= 0).*dGamma./((abs(k_n_l{l+1}))+(real(k_n_l{l+1}) == 0)));
        
        k_n_l_orig = k_n_l{l+1};
        k_n_l{l+1} = k_n_l_aug;
        
        S_l_k = exp(-2i*k_vec_out*a);
        d_delta_l_k_dk = -a;
        for k_ind=1:length(k_n_l{l+1})
            S_l_k = S_l_k.*(k_n_l{l+1}(k_ind)+k_vec_out)./(k_n_l{l+1}(k_ind)-k_vec_out);
            d_delta_l_k_dk = d_delta_l_k_dk + ...
                (-(imag(k_n_l{l+1}(k_ind)).*((imag(k_n_l{l+1}(k_ind))).^2 + k_vec_out.^2 + ...
                (real(k_n_l{l+1}(k_ind))).^2))./ ...
                (k_vec_out.^4 + 2*k_vec_out.^2.*(imag(k_n_l{l+1}(k_ind)) - real(k_n_l{l+1}(k_ind))).* ...
                (imag(k_n_l{l+1}(k_ind)) + real(k_n_l{l+1}(k_ind))) + ((imag(k_n_l{l+1}(k_ind))).^2 + ...
                (real(k_n_l{l+1}(k_ind))).^2).^2));
        end
        
        sigma_l(:,l+1) = (2*l+1)*(pi./k_vec_out.^2).*abs(1-S_l_k).^2;
        tau_l(:,l+1) = d_delta_l_k_dk./k_vec_out;
        S_l(:,l+1) = S_l_k;
    end
end
end