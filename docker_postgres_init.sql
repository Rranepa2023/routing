CREATE TABLE attributes1
(
    id_pk  serial PRIMARY KEY,
    ranepa_num      text,
    ranepa_dat      date,
    korres_num      text,
    korres_dat      date,
    ranepa_n_user   text,
    ranepa_d_user   date,
    korres_n_user   text,
    korres_d_user   date
);

alter table attributes1 owner to postgres;