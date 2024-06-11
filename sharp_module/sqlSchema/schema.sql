CREATE TABLE harp_definition (
    harp_number int NOT NULL UNIQUE,
    start_time timestamp without time zone,
    end_time timestamp without time zone,
    number_of_phenomena int,
    PRIMARY KEY (harp_number)
);


CREATE TABLE harp_phenomena (
    id SERIAL,
    NOAA_id int NOT NULL,
    harp_number int NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT harp_phenomena_harp_definition_harp_number
      FOREIGN KEY(harp_number)
      REFERENCES harp_definition(harp_number)
);

CREATE TABLE harp_evolution (
    id SERIAL,
    harp_number int NOT NULL,
    evolution_time timestamp without time zone NOT NULL,
    evolution_image bytea NOT NULL,
    usflux float,
    PRIMARY KEY (id),
    CONSTRAINT harp_evolution_harp_definition_harp_number
      FOREIGN KEY(harp_number)
      REFERENCES harp_definition(harp_number)
);